#include <string>
#include <iostream>

namespace ml
{


void warningFunctionMLIB(const std::string &description)
{
	std::cout << description << std::endl;
    //DEBUG_BREAK;
}

void errorFunctionMLIB(const std::string &description)
{
	std::cout << description << std::endl;
	DEBUG_BREAK;
}

void assertFunctionMLIB(bool statement, const std::string &description)
{
	if(!statement)
	{
		std::cout << description << std::endl;
#ifdef _DEBUG	
		DEBUG_BREAK;
#endif
	}
}

}  // namespace ml
