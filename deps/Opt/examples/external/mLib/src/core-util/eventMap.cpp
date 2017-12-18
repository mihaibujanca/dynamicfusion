
namespace ml
{

void EventMap::registerEvent(const std::string &event, const std::function<void(vector<std::string> &params)> &handler)
{
    _handlers[event] = handler;
}

void EventMap::dispatchEvents(const vector<std::string> &messages) const
{
    for (const std::string &message : messages)
    {
        auto parts = ml::util::split(message, ' ');
        if (_handlers.count(parts[0]) == 0)
        {
            std::cout << "No event handler for: " << message << std::endl;
        }
        else
        {
            _handlers.find(parts[0])->second(parts);
        }
    }
}

#ifdef _WIN32
void EventMap::dispatchEvents(ml::UIConnection &ui) const
{
	ui.readMessages();
	dispatchEvents(ui.messages());
	ui.messages().clear();
}
#endif

}