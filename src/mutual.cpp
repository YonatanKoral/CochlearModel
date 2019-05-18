
#include "mutual.h"

void pause()
{

	//Clear whatever's still in the buffer
    std::cout << "Press Enter to continue . . .\n";
	std::cin.ignore(MAX_BUF_SIZE, '\n');
	
}