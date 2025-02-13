#include "openai_api.h"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

using json = nlohmann::json;

// Write callback for libcurl
size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t totalSize = size * nmemb;
    std::string* str = static_cast<std::string*>(userp);
    str->append(static_cast<char*>(contents), totalSize);
    return totalSize;
}

std::string callChatGPTAPI(const std::string &prompt) {
    CURL *curl;
    CURLcode res;
    std::string readBuffer;
    curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("Failed to initialize curl");
    }
    
    // Set the URL for ChatGPT (OpenAI Chat Completions endpoint)
    curl_easy_setopt(curl, CURLOPT_URL, "https://api.openai.com/v1/chat/completions");
    
    // Set up HTTP headers.
    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    // Replace with your actual API key.
    headers = curl_slist_append(headers, "Authorization: Bearer YOUR_API_KEY");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    
    // Create the JSON payload.
    json j;
    j["model"] = "gpt-3.5-turbo";
    j["messages"] = json::array({
        {{"role", "system"}, {"content", "You are an expert image processing assistant."}},
        {{"role", "user"}, {"content", prompt}}
    });
    std::string payload = j.dump();
    
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
    
    res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        curl_easy_cleanup(curl);
        throw std::runtime_error("curl_easy_perform() failed: " + std::string(curl_easy_strerror(res)));
    }
    
    curl_easy_cleanup(curl);
    curl_slist_free_all(headers);
    
    auto responseJson = json::parse(readBuffer);
    std::string reply;
    try {
        reply = responseJson["choices"][0]["message"]["content"];
    } catch (...) {
        reply = "Error: Unexpected response format.";
    }
    
    return reply;
}