

namespace ml {
	//
	// In a multi-window world, s_windowMap would be needed.  In a single-window world, s_mainWindow is sufficient.
	//
	//std::map<HWND, WindowWin32*> s_windowMap;
	ml::WindowWin32* s_mainWindow = nullptr;

	LRESULT WINAPI WindowCallback(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
	{

		if (s_mainWindow == nullptr || !s_mainWindow->getParent().initialized()) return DefWindowProc(hWnd, msg, wParam, lParam);

		if (s_mainWindow->msgProcCallback != nullptr) {
				if (s_mainWindow->msgProcCallback(hWnd, msg, wParam, lParam)) { return 0; }  // Return if handled
		}
		auto &parent = s_mainWindow->getParent();

		switch (msg)
		{
		case WM_SYSCOMMAND:
		{
			switch (wParam)
			{
			case SC_SCREENSAVE:      // screensaver trying To start
			case SC_MONITORPOWER:    // monitor trying to enter powersave
				return 0;
			}
			break;
		}

		case WM_CLOSE:
			PostQuitMessage(0);
			break;

		case WM_KEYDOWN:
			switch (wParam)                 //key pressed
			{
			case VK_ESCAPE:
				//    PostQuitMessage(0);        //if escape pressed, exit
				//    break;
			default:
				UINT keyIndex = (UINT)wParam;
				parent.callback().keyDown(parent.data(), (UINT)wParam);
				if (keyIndex < ml::InputState::keyCount) parent.data().input.keys[keyIndex] = true;
				break;
			}
			break;

		case WM_KEYUP:
		{
			UINT keyIndex = (UINT)wParam;
			if (keyIndex < ml::InputState::keyCount) parent.data().input.keys[keyIndex] = false;
		}
		break;

		case WM_SIZE:
			parent.setResizeEvent();
			break;

		case WM_LBUTTONDOWN:
			parent.data().input.mouse.buttons[ml::MouseButtonLeft] = true;
			parent.callback().mouseDown(parent.data(), ml::MouseButtonLeft);
			break;

		case WM_LBUTTONUP:
			parent.data().input.mouse.buttons[ml::MouseButtonLeft] = false;
			break;

		case WM_RBUTTONDOWN:
			parent.data().input.mouse.buttons[ml::MouseButtonRight] = true;
			parent.callback().mouseDown(parent.data(), ml::MouseButtonRight);
			break;

		case WM_RBUTTONUP:
			parent.data().input.mouse.buttons[ml::MouseButtonRight] = false;
			break;

		case WM_MOUSEMOVE:
		{
			POINTS p = MAKEPOINTS(lParam);
			parent.data().input.prevMouse.pos = parent.data().input.mouse.pos;
			parent.data().input.mouse.pos = ml::vec2i(p.x, p.y);
			parent.callback().mouseMove(parent.data());

		}
		break;

		case WM_MOUSEWHEEL:
			parent.callback().mouseWheel(parent.data(), GET_WHEEL_DELTA_WPARAM(wParam));
			break;
		}

		return DefWindowProc(hWnd, msg, wParam, lParam);
	}

	ml::WindowWin32::~WindowWin32()
	{
		destroy();
		s_mainWindow = nullptr;
	}



	void ml::WindowWin32::init(HINSTANCE instance, int width, int height, const std::string &name, MsgProcCallback msgProcFun /* = nullptr*/, unsigned int initWindowPosX  /* = 0 */, unsigned int initWindowPosY  /* = 0*/)
	{
		// width/height need to be client width/height
		RECT wr = { 0, 0, width, height };
		AdjustWindowRect(&wr, WS_OVERLAPPEDWINDOW, false); // adjust window size

		m_className = name;
		auto className = util::windowsStr(m_className);

		if (msgProcFun != nullptr) { msgProcCallback = msgProcFun; }

		m_class.style = CS_HREDRAW | CS_VREDRAW;
		m_class.lpfnWndProc = (WNDPROC)WindowCallback;
		m_class.cbClsExtra = 0;
		m_class.cbWndExtra = 0;
		m_class.hInstance = instance;
		m_class.hIcon = LoadIcon(nullptr, IDI_WINLOGO);
		m_class.hCursor = LoadCursor(nullptr, IDC_ARROW);
		m_class.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
		m_class.lpszMenuName = nullptr;
		m_class.lpszClassName = className.c_str();
		if (RegisterClass(&m_class) == 0) {
			util::errorExit(__FUNCTION__);
		}

		s_mainWindow = this;
		m_handle = CreateWindow(
			className.c_str(),
			className.c_str(),
			WS_OVERLAPPEDWINDOW,
			initWindowPosX, //CW_USEDEFAULT
			initWindowPosY, //CW_USEDEFAULT
			wr.right - wr.left,    // width of the window
			wr.bottom - wr.top,    // height of the window
			(HWND) nullptr,
			(HMENU) nullptr,
			instance,
			(LPVOID) nullptr);
		if (m_handle == nullptr) {
			util::errorExit(__FUNCTION__);
		}
		ShowWindow(m_handle, SW_SHOW);
		UpdateWindow(m_handle);
	}

	void ml::WindowWin32::destroy()
	{
		DestroyWindow(m_handle);
		auto className = util::windowsStr(m_className);
		UnregisterClass(className.c_str(), m_class.hInstance);
	}

	UINT ml::WindowWin32::getWidth() const
	{
		RECT rect;
		GetClientRect(m_handle, &rect);
		return rect.right - rect.left;
	}

	UINT ml::WindowWin32::getHeight() const
	{
		RECT rect;
		GetClientRect(m_handle, &rect);
		return rect.bottom - rect.top;
	}

}
