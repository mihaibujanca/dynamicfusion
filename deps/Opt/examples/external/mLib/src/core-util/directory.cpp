
namespace ml {

Directory::Directory(const std::string& path)
{
    init(path);
}

std::vector<std::string> Directory::getFilesWithSuffix(const std::string& suffix) const
{
    std::vector<std::string> result;
    for (size_t fileIndex = 0; fileIndex < m_files.size(); fileIndex++)
    {
        const std::string& filename = m_files[fileIndex];
        if(util::endsWith(filename, suffix))
        {
            result.push_back(filename);
        }
    }
    return result;
}



std::vector<std::string> Directory::getFilesWithPrefix(const std::string& prefix) const
{
	std::vector<std::string> result;
	for (size_t fileIndex = 0; fileIndex < m_files.size(); fileIndex++)
	{
		const std::string& filename = m_files[fileIndex];
		if (util::startsWith(filename, prefix))
		{
			result.push_back(filename);
		}
	}
	return result;
}

std::vector<std::string> Directory::getFilesContaining(const std::string &str) const
{
	std::vector<std::string> result;
	for (size_t fileIndex = 0; fileIndex < m_files.size(); fileIndex++)
	{
		const std::string& filename = m_files[fileIndex];
		if (util::contains(filename, str))
		{
			result.push_back(filename);
		}
	}
	return result;
}




std::vector<std::string> Directory::getDirectoriesWithSuffix(const std::string& suffix) const
{
	std::vector<std::string> result;
	for (size_t fileIndex = 0; fileIndex < m_directories.size(); fileIndex++)
	{
		const std::string& filename = m_directories[fileIndex];
		if (util::endsWith(filename, suffix))
		{
			result.push_back(filename);
		}
	}
	return result;
}



std::vector<std::string> Directory::getDirectoriesWithPrefix(const std::string& prefix) const
{
	std::vector<std::string> result;
	for (size_t fileIndex = 0; fileIndex < m_directories.size(); fileIndex++)
	{
		const std::string& filename = m_directories[fileIndex];
		if (util::startsWith(filename, prefix))
		{
			result.push_back(filename);
		}
	}
	return result;
}

std::vector<std::string> Directory::getDirectoriesContaining(const std::string &str) const
{
	std::vector<std::string> result;
	for (size_t fileIndex = 0; fileIndex < m_directories.size(); fileIndex++)
	{
		const std::string& filename = m_directories[fileIndex];
		if (util::contains(filename, str))
		{
			result.push_back(filename);
		}
	}
	return result;
}





#ifdef WIN32
void Directory::init(const std::string &path)
{
	m_path = path + "/";
	m_files.clear();
	m_directories.clear();

	WIN32_FIND_DATAA findResult;

	HANDLE hFind = FindFirstFileA((path + std::string("\\*")).c_str(), &findResult);

	if (hFind == INVALID_HANDLE_VALUE) return;

	do
	{
		if (findResult.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
		{
			std::string directoryName(findResult.cFileName);
			if(!util::startsWith(directoryName, "."))
			{
				m_directories.push_back(directoryName);
			}
		}
		else
		{
			//FileSize.LowPart = findResult.nFileSizeLow;
			//FileSize.HighPart = findResult.nFileSizeHigh;
			m_files.push_back(std::string(findResult.cFileName));
		}
	}
	while (FindNextFileA(hFind, &findResult) != 0);

	FindClose(hFind);
}
#endif  // _WIN32

#ifdef LINUX
void Directory::init(const std::string &path)
{
    //std::cout << "Loading all files in " << path << std::endl;
	m_path = path + "\\";
	m_files.clear();
	m_directories.clear();

	auto dir = opendir(path.c_str());
    if (dir == nullptr)
    {
        std::cout << "Could not open " << path << std::endl;
        return; // could not open directory
    }

	auto entity = readdir(dir);

	while (entity != nullptr) {
		auto entity = readdir(dir);
        //std::cout << "Reading entity " << entity << std::endl;
        if (entity == nullptr) break;
		if (entity->d_type == DT_DIR) {
			// don't process  '..' & '.' directories
			if(entity->d_name[0] != '.')
				m_directories.push_back(std::string(entity->d_name));
		}
		else if (entity->d_type == DT_REG) {
            //std::cout << "Reading file " << entity->d_name << std::endl;
			m_files.push_back(std::string(entity->d_name));
		}
	}
	
	closedir(dir);
}
#endif  // LINUX

}  // namespace ml