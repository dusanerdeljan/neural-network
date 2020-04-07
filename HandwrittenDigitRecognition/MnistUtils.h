#pragma once
#include <fstream>

int ReadInt(std::ifstream& in)
{
	unsigned char c1, c2, c3, c4;
	int i;
	in.read(reinterpret_cast<char*>(&i), sizeof(i));
	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;
	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

unsigned char ReadChar(std::ifstream& in)
{
	unsigned char temp;
	in.read(reinterpret_cast<char*>(&temp), sizeof(temp));
	return temp;
}
