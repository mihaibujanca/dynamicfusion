

#ifdef _WIN32

#include <windows.h>
#include <strsafe.h>

namespace ml {

	namespace util {
		std::string getLastErrorString()
		{
			//Get the error message, if any.
			DWORD errorMessageID = ::GetLastError();
			if (errorMessageID == 0)
				return std::string(); //No error message has been recorded

			LPSTR messageBuffer = nullptr;
			size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
				NULL, errorMessageID, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&messageBuffer, 0, NULL);

			std::string message(messageBuffer, size);

			//Free the buffer.
			LocalFree(messageBuffer);

			return message;
		}
		void errorExit(const std::string& functionName)
		{
			// Retrieve the system error message for the last-error code

			LPVOID lpMsgBuf;
			LPVOID lpDisplayBuf;
			DWORD dw = GetLastError();

			FormatMessage(
				FORMAT_MESSAGE_ALLOCATE_BUFFER |
				FORMAT_MESSAGE_FROM_SYSTEM |
				FORMAT_MESSAGE_IGNORE_INSERTS,
				NULL,
				dw,
				MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
				(LPTSTR)&lpMsgBuf,
				0, NULL);

			// Display the error message and exit the process

			auto lpszFunctionStr = util::windowsStr(functionName);
			lpDisplayBuf = (LPVOID)LocalAlloc(LMEM_ZEROINIT,
				(lstrlen((LPCTSTR)lpMsgBuf) + lstrlen((LPCTSTR)lpszFunctionStr.c_str()) + 40) * sizeof(TCHAR));
			StringCchPrintf((LPTSTR)lpDisplayBuf,
				LocalSize(lpDisplayBuf) / sizeof(TCHAR),
				TEXT("%s failed with error %d: %s"),
				lpszFunctionStr.c_str(), dw, lpMsgBuf);
			MessageBox(NULL, (LPCTSTR)lpDisplayBuf, TEXT("Error"), MB_OK);
			DEBUG_BREAK;

			LocalFree(lpMsgBuf);
			LocalFree(lpDisplayBuf);
			ExitProcess(dw);
		}
	}

}

#endif
