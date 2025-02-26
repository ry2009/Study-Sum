import os
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import textwrap
import random
import importlib
import re
from itertools import combinations
from collections import Counter

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Try to import the ML summarizer, but don't fail if it's not available
try:
    from app.services.ml_summary_service import summarize_text_ml, generate_content_summary as ml_generate_content_summary
    ml_summarizer_available = True
    print("ML-based summarizer is available")
except ImportError:
    ml_summarizer_available = False
    print("ML-based summarizer is not available, using fallback methods")

def combine_summaries(summaries):
    """Combine multiple summaries into a unified format with better organization"""
    combined_text = ""
    
    # Organize by source type
    video_summaries = []
    document_summaries = []
    
    for summary in summaries:
        if summary.get('type') == 'video':
            video_summaries.append(summary)
        else:
            document_summaries.append(summary)
    
    # Add document summaries first
    if document_summaries:
        combined_text += "# Document Sources\n\n"
        for i, summary in enumerate(document_summaries):
            combined_text += f"## {summary.get('title', f'Document {i+1}')}\n\n"
            if 'summary' in summary:
                combined_text += f"{summary['summary']}\n\n"
            elif 'text' in summary:
                combined_text += f"{summary['text'][:300]}...\n\n"
            
            if 'key_terms' in summary and summary['key_terms']:
                combined_text += "Key terms: " + ", ".join(summary['key_terms']) + "\n\n"
    
    # Add video summaries
    if video_summaries:
        combined_text += "# Video Sources\n\n"
        for i, summary in enumerate(video_summaries):
            combined_text += f"## {summary.get('title', f'Video {i+1}')}\n\n"
            if 'summary' in summary:
                combined_text += f"{summary['summary']}\n\n"
    
    return combined_text

def extract_concepts(text):
    """Extract key concepts from text using NLP techniques"""
    # Tokenize and clean text
    sentences = sent_tokenize(text.lower())
    
    # Get all bigrams and trigrams
    phrases = []
    for sentence in sentences:
        # Clean sentence
        clean_sentence = re.sub(r'[^\w\s]', '', sentence)
        words = clean_sentence.split()
        
        # Get bigrams and trigrams
        for i in range(len(words) - 1):
            phrases.append(' '.join(words[i:i+2]))
        
        for i in range(len(words) - 2):
            phrases.append(' '.join(words[i:i+3]))
    
    # Count frequencies
    phrase_counts = Counter(phrases)
    
    # Filter out common words and too short phrases
    stop_words = set(nltk.corpus.stopwords.words('english'))
    filtered_phrases = {}
    for phrase, count in phrase_counts.items():
        words = phrase.split()
        # Check if all words in the phrase are not stop words
        if not all(word in stop_words for word in words) and len(phrase) > 5:
            filtered_phrases[phrase] = count
    
    # Return the most common phrases
    top_phrases = sorted(filtered_phrases.items(), key=lambda x: x[1], reverse=True)[:15]
    return [phrase for phrase, _ in top_phrases]

def generate_learning_points(topic, summaries):
    """Generate structured learning points from the source materials"""
    combined_text = combine_summaries(summaries)
    
    # Extract key sentences
    sentences = sent_tokenize(combined_text)
    
    # Create structured learning points
    learning_points = {
        "core_concepts": [],
        "key_principles": [],
        "applications": [],
        "examples": []
    }
    
    # Use a more sophisticated approach to extract concepts
    concepts = extract_concepts(combined_text)
    
    # Use clustering to group related concepts
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
        
        # Create a list of all sentences with their text
        sentence_texts = [s for s in sentences if len(s.split()) > 5]
        
        if len(sentence_texts) > 10:
            # Create TF-IDF matrix
            vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
            tfidf_matrix = vectorizer.fit_transform(sentence_texts)
            
            # Cluster the sentences 
            num_clusters = min(5, len(sentence_texts) // 2)
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            kmeans.fit(tfidf_matrix)
            
            # Get the sentences closest to the cluster centers
            order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
            terms = vectorizer.get_feature_names_out()
            
            # Extract key concepts from each cluster
            for i in range(min(3, num_clusters)):
                concept_terms = [terms[ind] for ind in order_centroids[i, :5]]
                concept_name = " ".join(concept_terms[:2])
                learning_points["core_concepts"].append(f"**{concept_name.title()}**: One of the primary concepts in {topic}")
    except:
        # Fall back to simpler method if ML approach fails
        for concept in concepts[:3]:
            learning_points["core_concepts"].append(f"**{concept.title()}**: One of the primary concepts in {topic}")
    
    # Ensure we have at least 3 core concepts
    while len(learning_points["core_concepts"]) < 3:
        if len(learning_points["core_concepts"]) == 0:
            learning_points["core_concepts"].append(f"**{topic.title()}**: The main subject of study")
        elif len(learning_points["core_concepts"]) == 1:
            learning_points["core_concepts"].append(f"**Applications of {topic}**: Practical ways to use {topic}")
        else:
            learning_points["core_concepts"].append(f"**Understanding {topic}**: Key aspects to comprehend")
    
    # Find sentences that might describe principles (sentences with should, must, important, etc.)
    principles = []
    principle_words = ["should", "must", "important", "key", "essential", "fundamental", "critical", "necessary", "required"]
    for sentence in sentences:
        for word in principle_words:
            if word in sentence.lower() and len(sentence) < 200:
                principles.append(sentence)
                break
    
    # Add principles section
    if principles:
        learning_points["key_principles"] = principles[:3]
    else:
        # Create more specific principles
        learning_points["key_principles"] = [
            f"Understanding the relationship between different aspects of {topic} is essential.",
            f"A methodical approach should be used when working with {topic}.",
            f"Practice and application are fundamental to mastering {topic}."
        ]
    
    # Find application sentences (often contain "used for", "applied to", "in practice", etc.)
    applications = []
    application_phrases = ["used for", "applied to", "in practice", "application", "implement", "useful in", "utilized for", "helps with"]
    for sentence in sentences:
        for phrase in application_phrases:
            if phrase in sentence.lower() and len(sentence) < 200:
                applications.append(sentence)
                break
    
    # Add applications section
    if applications:
        learning_points["applications"] = applications[:3]
    else:
        # Create more specific applications
        learning_points["applications"] = [
            f"{topic} can be applied in various fields including research and industry.",
            f"Practical implementation of {topic} requires understanding of its underlying principles.",
            f"The concepts of {topic} can be extended to solve real-world problems."
        ]
    
    # Example sentences - try to find actual examples from the text
    example_sentences = []
    example_phrases = ["for example", "for instance", "such as", "like", "e.g.", "specifically", "namely"]
    for sentence in sentences:
        for phrase in example_phrases:
            if phrase in sentence.lower() and len(sentence) < 250:
                example_sentences.append(sentence)
                break
    
    if example_sentences:
        learning_points["examples"] = example_sentences[:2]
    else:
        # Use concept sentences as examples if we couldn't find explicit examples
        concept_sentences = []
        for concept in concepts[:2]:
            for sentence in sentences:
                if concept in sentence.lower() and len(sentence) < 200:
                    concept_sentences.append(sentence)
                    break
                    
        learning_points["examples"] = concept_sentences if concept_sentences else [f"Example: Consider how {topic} applies in different contexts."]
    
    return learning_points

def generate_study_outline(topic, summaries):
    """Generate a comprehensive study outline based on the topic and summaries"""
    combined_text = combine_summaries(summaries)
    
    # Base structure for a study outline
    outline = {
        "introduction": {
            "title": f"Introduction to {topic}",
            "points": []
        },
        "core_content": {
            "title": f"Core Concepts of {topic}",
            "sections": []
        },
        "advanced_topics": {
            "title": f"Advanced Topics in {topic}",
            "points": []
        },
        "applications": {
            "title": f"Applications of {topic}",
            "points": []
        },
        "conclusion": {
            "title": f"Mastering {topic}",
            "points": []
        }
    }
    
    # Extract concepts for structuring the outline
    concepts = extract_concepts(combined_text)
    
    # Try using an NLP approach to structure the guide
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Tokenize and clean sentences
        sentences = sent_tokenize(combined_text)
        cleaned_sentences = []
        for sentence in sentences:
            # Clean up the sentence (remove special chars, etc.)
            clean_sentence = re.sub(r'[^\w\s]', '', sentence)
            if len(clean_sentence.split()) > 5:  # Only consider meaningful sentences
                cleaned_sentences.append(sentence)
        
        # Create vector representation of sentences
        vectorizer = TfidfVectorizer(stop_words='english')
        if len(cleaned_sentences) > 5:
            sentence_vectors = vectorizer.fit_transform(cleaned_sentences)
            
            # Get important words
            feature_names = vectorizer.get_feature_names_out()
            
            # For each concept, find the most relevant sentences
            for concept in concepts[:3]:
                concept_tfidf = vectorizer.transform([concept])
                
                # Calculate similarity with each sentence
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(concept_tfidf, sentence_vectors)
                
                # Get the top 2 sentences for this concept
                top_indices = similarities[0].argsort()[-2:][::-1]
                concept_sentences = [cleaned_sentences[idx] for idx in top_indices]
                
                # Create a section for this concept
                section = {
                    "title": concept.title(),
                    "points": concept_sentences
                }
                outline["core_content"]["sections"].append(section)
    except:
        # Fall back to the standard method if ML approach fails
        for concept in concepts[:3]:
            # Find sentences related to this concept
            concept_sentences = []
            for sentence in sent_tokenize(combined_text):
                if concept.lower() in sentence.lower() and len(sentence) < 200:
                    concept_sentences.append(sentence)
                    if len(concept_sentences) >= 2:
                        break
            
            # If we didn't find enough sentences, add a generic one
            if len(concept_sentences) < 2:
                concept_sentences.append(f"{concept.title()} is a fundamental component of {topic}.")
                
            # Create a section for this concept
            section = {
                "title": concept.title(),
                "points": concept_sentences
            }
            outline["core_content"]["sections"].append(section)
    
    # Make sure we have at least 3 core content sections
    while len(outline["core_content"]["sections"]) < 3:
        generic_titles = ["Key Components", "Fundamental Processes", "Theoretical Framework"]
        title = generic_titles[len(outline["core_content"]["sections"]) % len(generic_titles)]
        
        section = {
            "title": title,
            "points": [
                f"{title} are essential elements of {topic}.",
                f"Understanding {title.lower()} requires careful study and practice."
            ]
        }
        outline["core_content"]["sections"].append(section)
    
    # Introduction points - find definitional or introductory sentences
    intro_sentences = []
    intro_phrases = ["defined as", "refers to", "is a", "describes", "consists of", "can be understood as"]
    
    for sentence in sent_tokenize(combined_text):
        lower_s = sentence.lower()
        if topic.lower() in lower_s and any(phrase in lower_s for phrase in intro_phrases):
            intro_sentences.append(sentence)
            if len(intro_sentences) >= 3:
                break
    
    if intro_sentences:
        outline["introduction"]["points"] = intro_sentences[:3]
    else:
        outline["introduction"]["points"] = [
            f"{topic} is a key area that integrates various concepts and techniques.",
            f"Understanding the fundamentals of {topic} is essential for advanced study.",
            f"{topic} encompasses multiple interconnected principles and methods."
        ]
    
    # Advanced topics - look for complex terminology and challenging concepts
    advanced_indicators = ["advanced", "complex", "sophisticated", "challenging", "difficult", 
                          "in-depth", "detailed", "specialized", "expert", "higher-level"]
    advanced_sentences = []
    
    for sentence in sent_tokenize(combined_text):
        if any(indicator in sentence.lower() for indicator in advanced_indicators):
            advanced_sentences.append(sentence)
            if len(advanced_sentences) >= 3:
                break
    
    if advanced_sentences:
        outline["advanced_topics"]["points"] = advanced_sentences[:3]
    else:
        outline["advanced_topics"]["points"] = [
            f"Advanced aspects of {topic} build upon the core foundations.",
            f"Complex scenarios in {topic} require integrating multiple concepts.",
            f"Further study of {topic} reveals deeper patterns and relationships."
        ]
    
    # Applications - find real-world applications and use cases
    application_phrases = ["used for", "applied to", "in practice", "implement", "application", 
                          "real-world", "industry", "practical", "case study", "example"]
    application_sentences = []
    
    for sentence in sent_tokenize(combined_text):
        if any(phrase in sentence.lower() for phrase in application_phrases):
            application_sentences.append(sentence)
            if len(application_sentences) >= 3:
                break
    
    if application_sentences:
        outline["applications"]["points"] = application_sentences[:3]
    else:
        outline["applications"]["points"] = [
            f"{topic} has numerous practical applications across different domains.",
            f"The principles of {topic} can be applied to solve complex problems.",
            f"Real-world implementations of {topic} demonstrate its practical value."
        ]
    
    # Conclusion - tips for mastery and further study
    outline["conclusion"]["points"] = [
        f"Mastering {topic} requires both theoretical knowledge and practical experience.",
        f"Regular practice and application of {topic} concepts leads to proficiency.",
        f"Continuing education in {topic} will reveal new insights and techniques."
    ]
    
    return outline

def generate_educational_slide(topic, learning_points, filename="study_slide.png"):
    """Generate an educational slide with proper layout and visual design"""
    try:
        # Create a slide with better design
        width, height = 1200, 800
        
        # Create the image with a gradient background
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        # Draw a gradient background
        for y in range(height):
            # Create a subtle blue gradient
            r = int(235 - y * 30 / height)
            g = int(245 - y * 30 / height)
            b = int(255 - y * 20 / height)
            for x in range(width):
                draw.point((x, y), fill=(r, g, b))
        
        # Draw a header bar
        draw.rectangle([(0, 0), (width, 80)], fill=(41, 128, 185))
        
        # Try to load fonts
        try:
            title_font = ImageFont.truetype("Arial Bold", 40)
            header_font = ImageFont.truetype("Arial Bold", 28)
            body_font = ImageFont.truetype("Arial", 22)
        except:
            # Fall back to default fonts
            title_font = ImageFont.load_default()
            header_font = title_font
            body_font = title_font
        
        # Draw title
        draw.text((width/2, 40), topic.upper(), fill="white", font=title_font, anchor="mm")
        
        # Add core concepts section
        section_y = 100
        draw.text((40, section_y), "Core Concepts", fill=(41, 128, 185), font=header_font)
        section_y += 40
        
        # Add concept points
        for i, concept in enumerate(learning_points["core_concepts"][:3]):
            wrapped_text = textwrap.fill(concept, width=50)
            for line in wrapped_text.split('\n'):
                draw.text((60, section_y), "• " + line, fill="black", font=body_font)
                section_y += 30
            section_y += 10
        
        # Add key principles section
        section_y += 20
        draw.text((40, section_y), "Key Principles", fill=(41, 128, 185), font=header_font)
        section_y += 40
        
        # Add principle points
        for i, principle in enumerate(learning_points["key_principles"][:2]):
            wrapped_text = textwrap.fill(principle, width=50)
            for line in wrapped_text.split('\n'):
                draw.text((60, section_y), "• " + line, fill="black", font=body_font)
                section_y += 30
            section_y += 10
        
        # Add applications section on the right
        right_section_x = width // 2 + 20
        right_section_y = 100
        draw.text((right_section_x, right_section_y), "Applications", fill=(41, 128, 185), font=header_font)
        right_section_y += 40
        
        # Add application points
        for i, application in enumerate(learning_points["applications"][:2]):
            wrapped_text = textwrap.fill(application, width=50)
            for line in wrapped_text.split('\n'):
                draw.text((right_section_x + 20, right_section_y), "• " + line, fill="black", font=body_font)
                right_section_y += 30
            right_section_y += 10
        
        # Add examples section
        right_section_y += 20
        draw.text((right_section_x, right_section_y), "Examples", fill=(41, 128, 185), font=header_font)
        right_section_y += 40
        
        # Add example points
        for i, example in enumerate(learning_points["examples"]):
            wrapped_text = textwrap.fill(example, width=50)
            for line in wrapped_text.split('\n'):
                draw.text((right_section_x + 20, right_section_y), "• " + line, fill="black", font=body_font)
                right_section_y += 30
            right_section_y += 10
        
        # Add a footer
        draw.rectangle([(0, height-40), (width, height)], fill=(41, 128, 185, 180))
        draw.text((width/2, height-20), "Study Guide", fill="white", font=body_font, anchor="mm")
        
        # Save the image
        os.makedirs("app/static", exist_ok=True)
        output_path = os.path.join("app/static", filename)
        image.save(output_path)
        
        # Create a full URL path for the template
        url_path = f"/static/{filename}"
        return url_path
    except Exception as e:
        raise Exception(f"Error generating educational slide: {str(e)}")

def generate_detailed_summary(sources, topic):
    """Generate a detailed educational summary and study materials from multiple sources"""
    try:
        # If ML summarizer is available, use it
        if ml_summarizer_available:
            try:
                ml_result = ml_generate_content_summary(sources, topic)
                
                # Generate learning points
                learning_points = generate_learning_points(topic, sources)
                
                # Generate study outline
                study_outline = generate_study_outline(topic, sources)
                
                # Generate an educational slide
                image_path = generate_educational_slide(topic, learning_points, f"{topic.replace(' ', '_')}_study.png")
                
                # Combine ML results with enhanced educational content
                return {
                    "topic": topic,
                    "study_outline": study_outline,
                    "learning_points": learning_points,
                    "image_path": image_path,
                    "combined_summary": ml_result['summary'],
                    "ai_suggestions": [
                        f"Focus on understanding the core concepts of {topic} first",
                        f"Practice applying {topic} principles to real-world examples",
                        f"Create a concept map connecting the ideas in {topic}",
                        f"Teach {topic} concepts to someone else to reinforce your understanding"
                    ]
                }
            except Exception as e:
                print(f"Error using ML summarizer: {e}. Falling back to standard method.")
                # Fall back to standard method if ML fails
        
        # Generate learning points using standard methods
        learning_points = generate_learning_points(topic, sources)
        
        # Generate study outline
        study_outline = generate_study_outline(topic, sources)
        
        # Generate an educational slide
        image_path = generate_educational_slide(topic, learning_points, f"{topic.replace(' ', '_')}_study.png")
        
        # Final study materials
        return {
            "topic": topic,
            "study_outline": study_outline,
            "learning_points": learning_points,
            "image_path": image_path, 
            "combined_summary": combine_summaries(sources),
            "ai_suggestions": [
                f"Focus on understanding the core concepts of {topic} first",
                f"Practice applying {topic} principles to real-world examples",
                f"Create a concept map connecting the ideas in {topic}",
                f"Teach {topic} concepts to someone else to reinforce your understanding"
            ]
        }
    except Exception as e:
        raise Exception(f"Error generating educational materials: {str(e)}") 