import os
import csv
import ast
import tempfile
import whisper
from pydub import AudioSegment

def load_whisper_model(model_name: str = "base"):
    return whisper.load_model(model_name)

def extract_audio_range(audio_path, start, end):
    audio = AudioSegment.from_wav(audio_path)
    segment = audio[start * 1000:end * 1000]  # convert to milliseconds
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    segment.export(temp_file.name, format="wav")
    return temp_file.name

def parse_timestamp(ts):
    if isinstance(ts, (float, int)):
        return float(ts)
    if isinstance(ts, str) and ":" in ts:
        h, m, s = ts.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    return float(ts)

def transcribe_audio(model, audio_path: str, speech_ranges=None) -> dict:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if not speech_ranges:
        return whisper.transcribe(model, audio_path)

    all_segments = []
    for start, end in speech_ranges:
        if end - start < 1.0:
            print(f"Skipped short segment: {start}-{end} (less than 1 second)")
            continue
        temp_path = extract_audio_range(audio_path, start, end)
        try:
            partial_result = whisper.transcribe(
                model, temp_path,
                condition_on_previous_text=False,
                no_speech_threshold=0.0
            )
            for seg in partial_result.get("segments", []):
                seg["start"] += start
                seg["end"] += start
            all_segments.extend(partial_result.get("segments", []))
        except Exception as e:
            print(f"Error transcribing segment {start}-{end} of {audio_path}: {e}")
        finally:
            os.remove(temp_path)

    return {"segments": all_segments}

def process_dataset(dataset_path: str, model, input_csv: str, output_csv: str) -> None:
    with open(input_csv, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        rows = list(reader)

    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header + ["Timestamps", "Texts"])

        for row in rows:
            video_file_name = row[0].replace(".mp4", ".wav")
            video_hate_speech = row[1]
            speech_ranges_str = row[5]
            print(f"Speech ranges: {speech_ranges_str} for {video_file_name}")
            print(f"Row data: {row}")

            try:
                if not speech_ranges_str.strip():
                    raise ValueError("Empty speech range")
                raw_ranges = ast.literal_eval(speech_ranges_str)
                speech_ranges = [(parse_timestamp(start), parse_timestamp(end)) for start, end in raw_ranges]
            except Exception as e:
                print(f"Invalid speech_ranges for {video_file_name}: {speech_ranges_str} — {e}")
                continue

            folder = "hate_audios_clean" if video_hate_speech == "Hate" else "non_hate_audios_clean"
            audio_path = os.path.abspath(os.path.join(dataset_path, folder, video_file_name))
            print(f"Processing: {audio_path}")

            try:
                result = transcribe_audio(model, audio_path, speech_ranges)
                segments = result.get("segments", [])

                timestamps = [[
                    f"{int(seg['start'] // 3600):02}:{int((seg['start'] % 3600) // 60):02}:{int(seg['start'] % 60):02}",
                    f"{int(seg['end'] // 3600):02}:{int((seg['end'] % 3600) // 60):02}:{int(seg['end'] % 60):02}"
                ] for seg in segments]

                texts = [seg.get("text", "") for seg in segments]
                writer.writerow(row + [timestamps, texts])

            except Exception as e:
                print(f"Error processing {video_file_name}: {e}")

    print(f"Transcription results saved to {output_csv}")


def speech_ranges_to_timestamps(audio_path, speech_ranges, model_name="base"):
    """
    Transcribe only the specified speech_ranges from the given WAV file
    and return aligned timestamps and texts.

    Args:
        audio_path (str): Path to the .wav audio file.
        speech_ranges (list of tuple): List of (start, end) times in seconds or "HH:MM:SS" strings.
        model_name (str): Whisper model size to load (default "base").

    Returns:
        timestamps (list of [str, str]): List of [start_ts, end_ts] strings "HH:MM:SS".
        texts (list of str): List of transcribed text for each segment.
    """
    # load model
    model = load_whisper_model(model_name)

    # parse any string timestamps into floats
    parsed_ranges = [
        (parse_timestamp(start), parse_timestamp(end))
        for start, end in speech_ranges
    ]

    # run transcription on each segment
    result = transcribe_audio(model, audio_path, parsed_ranges)
    segments = result.get("segments", [])

    # format output
    timestamps = [
        [
            f"{int(seg['start'] // 3600):02}:{int((seg['start'] % 3600) // 60):02}:{int(seg['start'] % 60):02}",
            f"{int(seg['end']   // 3600):02}:{int((seg['end']   % 3600) // 60):02}:{int(seg['end']   % 60):02}"
        ]
        for seg in segments
    ]
    texts = [seg.get("text", "").strip() for seg in segments]

    return timestamps, texts


def main():
    model = load_whisper_model("base")
    dataset_path = "dataset"
    input_csv = "HateMM_annotation_speech_process.csv"
    output_csv = "transcription_results.csv"
    process_dataset(dataset_path, model, input_csv, output_csv)

if __name__ == "__main__":
    main()