import re
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import requests
import json
import os
from datetime import datetime

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def extract_video_id(url):
    """Extract YouTube video ID from URL"""
    youtube_regex = (
        r'(https?://)?(www\.)?'
        '(youtube|youtu|youtube-nocookie)\.(com|be)/'
        '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')
    
    youtube_match = re.match(youtube_regex, url)
    if youtube_match:
        return youtube_match.group(6)
    return None

def get_video_info_fallback(video_id):
    """Fallback method to get video information using public API"""
    try:
        # Try using YouTube's oEmbed API - doesn't require API key
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        response = requests.get(oembed_url)
        if response.status_code == 200:
            data = response.json()
            return {
                "title": data.get("title", "Unknown Title"),
                "author": data.get("author_name", "Unknown Author"),
                "length": 0,  # Not available in oembed
                "thumbnail_url": f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
                "views": 0,  # Not available in oembed
                "publish_date": str(datetime.now().date()),  # Not available in oembed
                "description": "Description not available"  # Not available in oembed
            }
    except Exception as e:
        print(f"Fallback API error: {str(e)}")
    
    # Return minimal info if all else fails
    return {
        "title": "Video Title Unavailable",
        "author": "Unknown Author",
        "length": 0,
        "thumbnail_url": f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
        "views": 0,
        "publish_date": str(datetime.now().date()),
        "description": "Description not available"
    }

def extract_video_info(url):
    """Extract basic information about the video"""
    try:
        video_id = extract_video_id(url)
        if not video_id:
            raise ValueError("Invalid YouTube URL")
        
        try:
            # Try using pytube
            yt = YouTube(url)
            
            return {
                "title": yt.title,
                "author": yt.author,
                "length": yt.length,
                "thumbnail_url": yt.thumbnail_url,
                "views": yt.views,
                "publish_date": str(yt.publish_date) if yt.publish_date else None,
                "description": yt.description
            }
        except Exception as pytube_error:
            print(f"PyTube error: {str(pytube_error)}")
            # If pytube fails, use fallback method
            return get_video_info_fallback(video_id)
    except Exception as e:
        raise Exception(f"Error extracting video info: {str(e)}")

def get_transcript(video_id):
    """Get the transcript of a YouTube video"""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = ' '.join([item['text'] for item in transcript_list])
        return transcript_text
    except NoTranscriptFound:
        return None
    except Exception as e:
        raise Exception(f"Error getting transcript: {str(e)}")

def sentence_similarity(sent1, sent2, stopwords=None):
    """Calculate similarity between two sentences."""
    if stopwords is None:
        stopwords = []
    
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    
    all_words = list(set(sent1 + sent2))
    
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    # Build the vectors for both sentences
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
    
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
    
    # Calculate cosine similarity
    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):
    """Build a similarity matrix for all sentences."""
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    # Calculate similarity for each pair of sentences
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i == j:  # Same sentence, similarity is 1
                similarity_matrix[i][j] = 1
            else:
                similarity_matrix[i][j] = sentence_similarity(
                    sentences[i], sentences[j], stop_words)
    
    return similarity_matrix

def extract_summary_from_text(text, num_sentences=5):
    """Extract summary using enhanced extractive summarization."""
    if not text:
        return "No text available for summarization."
    
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Return the whole text if it's already shorter than requested
    if len(sentences) <= num_sentences:
        return text
    
    # Get stop words
    stop_words = set(stopwords.words('english'))
    
    # Tokenize each sentence into words
    sentence_tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
    
    # Build the similarity matrix
    similarity_matrix = build_similarity_matrix(sentence_tokens, stop_words)
    
    # Create a graph and apply pagerank
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    
    # Sort sentences by score
    ranked_sentences = sorted([(scores[i], sentences[i]) for i in range(len(sentences))], reverse=True)
    
    # Get top sentences
    top_sentences = [ranked_sentences[i][1] for i in range(min(num_sentences, len(ranked_sentences)))]
    
    # Sort by their original order
    ordered_sentences = []
    for sentence in sentences:
        if sentence in top_sentences:
            ordered_sentences.append(sentence)
            if len(ordered_sentences) == num_sentences:
                break
    
    # Join the sentences
    summary = ' '.join(ordered_sentences)
    
    # If the summary is still too long, compress it further by removing less important words
    if len(summary.split()) > 100:
        # Remove stopwords from the summary
        words = summary.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        # Join back but ensure we keep some structure
        compressed_summary = ' '.join(filtered_words)
        # If we've compressed it enough, use it
        if len(compressed_summary) < len(summary) * 0.8:
            return compressed_summary
    
    return summary

def summarize_text(text, max_length=800):
    """Summarize text using extractive summarization with enhanced compression."""
    if not text:
        return "No transcript available for summarization."
    
    # Try to use ML-based summarization if available
    try:
        from app.services.ml_summary_service import summarize_text_ml
        return summarize_text_ml(text, max_length)
    except ImportError:
        # Fall back to the improved extractive method
        pass
    
    try:
        # Identify and remove common YouTube filler phrases
        filler_phrases = [
            "don't forget to like and subscribe",
            "please subscribe to the channel",
            "hit the like button",
            "in this video",
            "in today's video",
            "in the description below",
            "welcome to this video",
            "welcome back to the channel",
            "in this tutorial",
            "thank you for watching"
        ]
        
        cleaned_text = text
        for phrase in filler_phrases:
            cleaned_text = re.sub(r'(?i)' + re.escape(phrase), '', cleaned_text)
        
        # First chunk the text if it's too long
        max_chunk_size = 10000
        summary = ""
        
        if len(cleaned_text) > max_chunk_size:
            # Split into meaningful chunks based on natural breaks
            chunks = []
            current_chunk = ""
            sentences = sent_tokenize(cleaned_text)
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < max_chunk_size:
                    current_chunk += " " + sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence
            
            if current_chunk:
                chunks.append(current_chunk)
                
            # Process each chunk to extract key information
            summaries = []
            for chunk in chunks:
                if len(chunk) > 100:  # Only process meaningful chunks
                    # Use fewer sentences for each chunk to make it more concise
                    chunk_summary = extract_summary_from_text(chunk, num_sentences=2)
                    summaries.append(chunk_summary)
            
            # Combine summaries and eliminate redundancy
            combined_text = ' '.join(summaries)
            
            # If still too long, summarize again with stricter settings
            if len(combined_text) > max_length:
                # Use a more aggressive approach for final summarization
                sentences = sent_tokenize(combined_text)
                stop_words = set(stopwords.words('english'))
                
                # Score sentences based on important words
                important_words = []
                for sentence in sentences:
                    words = nltk.word_tokenize(sentence.lower())
                    for word in words:
                        if word not in stop_words and len(word) > 3 and word.isalnum():
                            important_words.append(word)
                
                # Get word frequency
                from collections import Counter
                word_freq = Counter(important_words)
                
                # Get top keywords
                keywords = [word for word, count in word_freq.most_common(10)]
                
                # Score sentences based on keywords
                sentence_scores = []
                for i, sentence in enumerate(sentences):
                    score = 0
                    for keyword in keywords:
                        if keyword in sentence.lower():
                            score += 1
                    
                    # Boost score for intro and conclusion sentences
                    if i < 2:  # Intro sentences
                        score *= 1.5
                    elif i > len(sentences) - 3:  # Conclusion sentences
                        score *= 1.3
                        
                    sentence_scores.append((score, sentence))
                
                # Sort by score and take top sentences
                top_sentences = sorted(sentence_scores, key=lambda x: x[0], reverse=True)[:max(3, max_length//200)]
                
                # Reorganize sentences in original order
                ordered_top_sentences = []
                for sentence in sentences:
                    if any(sentence == s for _, s in top_sentences):
                        ordered_top_sentences.append(sentence)
                        if len(ordered_top_sentences) >= max(3, max_length//200):
                            break
                
                summary = ' '.join(ordered_top_sentences)
            else:
                summary = combined_text
        else:
            # For shorter text, more aggressive summarization
            sentences = sent_tokenize(cleaned_text)
            
            # Get only essential sentences (fewer for shorter texts)
            summary = extract_summary_from_text(cleaned_text, num_sentences=max(2, min(3, max_length//300)))
        
        # Post-process the summary to make it more readable
        sentences = sent_tokenize(summary)
        if len(sentences) > 1:
            # Add transitions between sentences to create a more coherent summary
            transitions = ["Additionally, ", "Moreover, ", "Furthermore, ", "Also, ", "In addition, "]
            enhanced_sentences = [sentences[0]]
            
            for i in range(1, len(sentences)):
                if i < len(transitions) and len(sentences[i].split()) > 3:
                    enhanced_sentences.append(transitions[i-1] + sentences[i][0].lower() + sentences[i][1:])
                else:
                    enhanced_sentences.append(sentences[i])
            
            summary = ' '.join(enhanced_sentences)
        
        return summary
    except Exception as e:
        return f"Error during summarization: {str(e)}. Using truncated text."

def summarize_video(url):
    """Extract and summarize content from a YouTube video"""
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")
    
    # Get transcript
    transcript = get_transcript(video_id)
    
    # If no transcript is available
    if not transcript:
        return {
            "transcript": "No transcript available for this video.",
            "summary": "No transcript available for summarization."
        }
    
    # Get video info for better context
    try:
        video_info = extract_video_info(url)
        video_title = video_info.get("title", "")
        video_author = video_info.get("author", "")
    except:
        video_title = ""
        video_author = ""
    
    # Summarize transcript
    summary = summarize_text(transcript)
    
    # Process summary to identify key points
    sentences = sent_tokenize(summary)
    
    # Format the summary with bullet points for readability
    if len(sentences) > 2:
        formatted_summary = "# Key points from this video:\n\n"
        
        # Add video title and author if available
        if video_title:
            formatted_summary += f"**Video: {video_title}**\n"
        if video_author:
            formatted_summary += f"**By: {video_author}**\n\n"
            
        # Convert to bullet points for better readability
        for i, sentence in enumerate(sentences):
            if i == 0:  # Keep first sentence as introduction
                formatted_summary += f"{sentence}\n\n"
            else:
                # Clean up the sentence and make it a bullet point
                clean_sentence = sentence.strip()
                if clean_sentence:
                    formatted_summary += f"• {clean_sentence}\n"
    else:
        # For very short summaries, keep as is but add headers
        formatted_summary = "# Key points from this video:\n\n"
        if video_title:
            formatted_summary += f"**Video: {video_title}**\n"
        if video_author:
            formatted_summary += f"**By: {video_author}**\n\n"
        formatted_summary += summary
    
    return {
        "transcript": transcript[:300] + "..." if transcript and len(transcript) > 300 else transcript,
        "summary": formatted_summary
    } 