import streamlit as st
import os
import tempfile
import gc
import base64
import time

from crewai import Agent, Crew, Process, Task
from crewai_tools import SerperDevTool
from src.tools.tool import DocumentSearchTool

