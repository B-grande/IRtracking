#ifndef OPENAI_API_H
#define OPENAI_API_H

#include <string>

// This function sends a prompt to the ChatGPT API and returns the response.
std::string callChatGPTAPI(const std::string &prompt);

#endif // OPENAI_API_H