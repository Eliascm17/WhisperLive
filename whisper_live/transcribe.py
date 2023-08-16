from requests import Session
from collections import deque
from dataclasses import dataclass

import pickle, struct, time, pyaudio
import threading
import os, json
import wave
import textwrap
import asyncio
import torch
import numpy as np

from fastapi.responses import HTMLResponse
from fastapi import APIRouter, Depends, Path, WebSocketDisconnect, Query, FastAPI, WebSocket
from whisper_live.transcriber import WhisperModel
from whisper_live.logger import get_logger
from whisper_live.utils import Singleton, get_connection_manager

logger = get_logger(__name__)

manager = get_connection_manager()

class Transcribe:
    RATE = 16000
    SERVER_READY = "SERVER_READY"

    def __init__(self, websocket: WebSocket, client_id: int, task="transcribe", device=None, multilingual=False, language=None):
        self.data = b""
        self.frames = b""
        self.language = language if multilingual else "en"
        self.task = task
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transcriber = WhisperModel(
            model_size_or_path="/app/whisper_model",
            device=device,
            compute_type="int8" if device=="cpu" else "float16",
            local_files_only=True
        )
        # voice activity detection model
        self.vad_model = torch.jit.load('/app/vad_model/silero_vad.jit')
        self.vad_threshold = 0.4        
        self.timestamp_offset = 0.0
        self.frames_np = None
        self.frames_offset = 0.0
        self.text = []
        self.current_out = ''
        self.prev_out = ''
        self.t_start=None
        self.exit = False
        self.same_output_threshold = 0
        self.show_prev_out_thresh = 5   # if pause(no output from whisper) show previous output for 5 seconds
        self.add_pause_thresh = 3       # add a blank to segment list as a pause(no speech) for 3 seconds
        self.transcript = []
        self.send_last_n_segments = 10

        self.clients = {}

        # text formatting
        self.wrapper = textwrap.TextWrapper(width=50)
        self.pick_previous_segments = 2

        self.websocket = websocket
        self.client_id = client_id

    async def speech_to_text(self):
        """
        Process audio stream in an infinite loop.
        """
        logger.info("Starting speech_to_text")

        while True:
            if self.exit:
                logger.info("Exiting speech to text thread")
                break

            if self.frames_np is None:
                logger.info("Frames are None, waiting for data")
                await asyncio.sleep(1)  # Add a sleep to prevent busy-waiting
                continue

            logger.info("Processing transcription")
            segments = await self.process_transcription(self.frames_np, self.frames_np.shape[0] / self.RATE)

            try:
                if len(segments) > 0:
                    logger.info(f"Sending {len(segments)} segments")
                    await self.send_segments(segments)
            except Exception as e:
                logger.error(f"[ERROR] Failed to send segments: {e}")
                await asyncio.sleep(0.01)

    async def process_transcription(self, input_bytes, duration):
        # clip audio if the current chunk exceeds 30 seconds
        if self.frames_np[int((self.timestamp_offset - self.frames_offset) * self.RATE):].shape[0] > 25 * self.RATE:
            self.timestamp_offset = self.frames_offset + (self.frames_np.shape[0] / self.RATE) - 5

        samples_take = max(0, (self.timestamp_offset - self.frames_offset) * self.RATE)
        input_bytes = self.frames_np[int(samples_take):].copy()
        duration = input_bytes.shape[0] / self.RATE
        if duration < 1.0:
            return []

        input_sample = input_bytes.copy()
        if len(self.text) and self.text[-1] != '':
            initial_prompt = self.text[-1]
        else:
            initial_prompt = None

        result = await asyncio.to_thread(
            self.transcriber.transcribe(
                input_sample,
                initial_prompt=initial_prompt,
                language=self.language,
                task=self.task
            ))

        segments = []
        if len(result):
            self.t_start = None
            last_segment = await self.update_segments(result, duration)
            if len(self.transcript) < self.send_last_n_segments:
                segments = self.transcript
            else:
                segments = self.transcript[-self.send_last_n_segments:]

            if last_segment is not None:
                segments = segments + [last_segment]
        else:
            if self.t_start is None:
                self.t_start = time.time()

            if time.time() - self.t_start < self.show_prev_out_thresh:
                if len(self.transcript) < self.send_last_n_segments:
                    segments = self.transcript
                else:
                    segments = self.transcript[-self.send_last_n_segments:]

            if len(self.text) and self.text[-1] != '':
                if time.time() - self.t_start > self.add_pause_thresh:
                    self.text.append('')

        return segments

    def fill_output(self, output):
        """
        Format output with current and previous complete segments
        into two lines of 50 characters.

        Args:
            output(str): current incomplete segment
        
        Returns:
            transcription wrapped in two lines
        """
        text = ''
        pick_prev = min(len(self.text), self.pick_previous_segments)
        for seg in self.text[-pick_prev:]:
            # discard everything before a 3 second pause
            if seg == '':
                text = ''
            else:
                text += seg
        wrapped = "".join(text + output)
        return wrapped
    
    def add_frames(self, frame_np):
        try:
            speech_prob = self.vad_model(torch.from_numpy(frame_np.copy()), self.RATE).item()
            if speech_prob < self.vad_threshold:
                return
            
        except Exception as e:
            logger.error(e)
            return
        
        if self.frames_np is not None and self.frames_np.shape[0] > 45*self.RATE:
            self.frames_offset += 30.0
            self.frames_np = self.frames_np[int(30*self.RATE):]
        if self.frames_np is None:
            self.frames_np = frame_np.copy()
        else:
            self.frames_np = np.concatenate((self.frames_np, frame_np), axis=0)

    def update_segments(self, segments, duration):
        """
        Processes the segments from whisper. Appends all the segments to the list
        except for the last segment assuming that it is incomplete.

        Args:
            segments(dict) : dictionary of segments as returned by whisper
            duration(float): duration of the current chunk
        
        Returns:
            transcription for the current chunk
        """
        offset = None
        self.current_out = ''
        last_segment = None
        # process complete segments
        if len(segments) > 1:
            for i, s in enumerate(segments[:-1]):
                text_ = s.text
                self.text.append(text_)
                start, end = self.timestamp_offset + s.start, self.timestamp_offset + min(duration, s.end)
                self.transcript.append(
                    {
                        'start': start,
                        'end': end,
                        'text': text_
                    }
                )
                
                offset = min(duration, s.end)

        self.current_out += segments[-1].text
        last_segment = {
            'start': self.timestamp_offset + segments[-1].start,
            'end': self.timestamp_offset + min(duration, segments[-1].end),
            'text': self.current_out
        }
        
        # if same incomplete segment is seen multiple times then update the offset
        # and append the segment to the list
        if self.current_out.strip() == self.prev_out.strip() and self.current_out != '': 
            self.same_output_threshold += 1
        else: 
            self.same_output_threshold = 0
        
        if self.same_output_threshold > 5:
            if not len(self.text) or self.text[-1].strip().lower()!=self.current_out.strip().lower():          
                self.text.append(self.current_out)
                self.transcript.append(
                    {
                        'start': self.timestamp_offset,
                        'end': self.timestamp_offset + duration,
                        'text': self.current_out
                    }
                )
            self.current_out = ''
            offset = duration
            self.same_output_threshold = 0
            last_segment = None
        else:
            self.prev_out = self.current_out
        
        # update offset
        if offset is not None:
            self.timestamp_offset += offset

        return last_segment

    async def send_segments(self, segments):
        try:
            segments_json = json.dumps(segments)
            await manager.send_message(message=segments_json, websocket=self.websocket)
        except Exception as e:
            logger.info(f"[ERROR]: {e}")
    
    def cleanup(self):
        logger.info("Cleaning up.")
        self.exit = True
        self.transcriber.destroy()

    async def initialize(self):
        await self.websocket.accept()
        await self.websocket.send_json(
            {
                "client_id": self.client_id,
                "message": self.SERVER_READY
            }
        )
