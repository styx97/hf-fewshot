stance_detection = {
"zero_shot" : """You are an expert in detecting the stance of a human utterance or a document towards a particular statement or topic. Given a user comment, output "favor" if the speaker expresses a favorable viewpoint or attitude towards the TOPIC. Output "against" if the stance is unfavorable or opposing, and "neutral" if the speaker's views neither favor or oppose the topic. If the document does not express any stance towards the topic at all, also respond with "neutral". You are only allowed to reply with "favor", "against", or "neutral", and output nothing after that.

TOPIC: {topic}
DOCUMENT: {text}
STANCE: """, 

"followup" : """
TOPIC: {topic}
DOCUMENT: {text}
STANCE: """
}