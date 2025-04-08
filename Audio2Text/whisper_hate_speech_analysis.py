import os
import csv
import whisper

def load_whisper_model(model_name: str = "base"):
    """
    Load the Whisper model.

    Args:
        model_name (str): The name of the Whisper model to load. Default is "base".

    Returns:
        Model: The loaded Whisper model.
    """
    return whisper.load_model(model_name)

def transcribe_audio(model, audio_path: str) -> dict:
    """
    Transcribe the audio file using the Whisper model.

    Args:
        model: The loaded Whisper model.
        audio_path (str): The path to the audio file.

    Returns:
        dict: The transcription result containing full text and segments.

    Raises:
        FileNotFoundError: If the audio file does not exist.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    result = whisper.transcribe(model, audio_path)
    return result

def process_dataset(dataset_path: str, model, input_csv: str, output_csv: str) -> None:
    """
    Process all audio files in the dataset directory and save transcription results.

    Args:
        dataset_path (str): Path to the dataset directory containing audio files.
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to the output CSV file.
        model: The loaded Whisper model.
    """
    # Read the input CSV file
    with open(input_csv, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)  # Read the header
        rows = list(reader)  # Read the remaining rows

    # Prepare the output CSV file
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Write the updated header
        writer.writerow(header + ["Timestamps", "Texts"])

        # Iterate over rows in the input CSV
        for row in rows:
            video_file_name = row[0]  # Assuming the first column is the video file name
            # Replace .mp4 by .wav
            video_file_name = video_file_name.replace(".mp4", ".wav")
            video_hate_speech = row[1]  # Assuming the second column is the hate speech label
            print(video_file_name, video_hate_speech)
            folder = "hate_audios_clean"
            if(video_hate_speech != "Hate"):
                folder = "non_hate_audios_clean"

            audio_path = os.path.join(dataset_path, folder, video_file_name)
            audio_path = os.path.abspath(audio_path)  # Get absolute path
            print(f"Processing: {audio_path}")

            try:
                # Transcribe the audio file
                result = transcribe_audio(model, audio_path)
                segments = result.get("segments", [])

                # Extract timestamps and texts
                timestamps = [[
                    f"{int(seg['start'] // 3600):02}:{int((seg['start'] % 3600) // 60):02}:{int(seg['start'] % 60):02}",
                    f"{int(seg['end'] // 3600):02}:{int((seg['end'] % 3600) // 60):02}:{int(seg['end'] % 60):02}"
                ] for seg in segments]
                texts = [seg.get("text", "") for seg in segments]

                # Write the updated row to the output CSV
                writer.writerow(row + [timestamps, texts])
            except Exception as e:
                print(f"Error processing {video_file_name}: {e}")

    print(f"Transcription results saved to {output_csv}")

def main():
    """
    Main function to process the dataset and save transcription results.
    """
    # Load the Whisper model
    model = load_whisper_model("base")

    # Define the dataset directory and output CSV file
    dataset_path = "dataset"
    input_csv = "HateMM_annotation_speech_process.csv"
    output_csv = "transcription_results.csv"

    # Process the dataset
    process_dataset(dataset_path, model, input_csv, output_csv)
if __name__ == "__main__":
    main()