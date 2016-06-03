/**
 * Some simple checks for the MyHashMap implementation
 */
#include <iostream>
#include <string>
#include <cassert>
#include "MyHashMap.h"

int main()
{
	if (MyHashMap::myHashFunction("0") != (48 % 29))
	{
		std::cout << "Error: myHashFunction bad return value for value 0." << std::endl;
		std::cout << "(see ~labcpp/www/ex1/HashSimpleCheck.cpp)" << std::endl;
		return 1;
	}
	if (MyHashMap::myHashFunction("00") != ((48 + 48) % 29))
	{
		std::cout << "Error: myHashFunction bad return value for value 00." << std::endl;
		std::cout << "(see ~labcpp/www/ex1/HashSimpleCheck.cpp)" << std::endl;
		return 1;
	}
	MyHashMap MyHashMap1, MyHashMap2;
	MyHashMap1.add("00", 1.);
	MyHashMap1.add("0", 2.);
	MyHashMap2.add("00", 1.);
	MyHashMap2.add("0", 3.);
	MyHashMap2.add("00", 0.3);
	double d;
	if (!MyHashMap2.isInHashMap("00", d))
	{
		std::cout << "Error: isInHashMap bad return value1." << std::endl;
		std::cout << "(see ~labcpp/www/ex1/HashSimpleCheck.cpp)" << std::endl;
		return 1;
	}
	else
	{
		if (d > 0.35 || d < 0.25)
		{
			std::cout << "Error: d should be 0.3, instead its value is " << d << std::endl;
			return 1;
		}
	}
	if (MyHashMap2.isInHashMap("000", d))
	{
		std::cout << "Error: isInHashMap bad return value2." << std::endl;
		std::cout << "(see ~labcpp/www/ex1/HashSimpleCheck.cpp)" << std::endl;
		return 1;
	}

	if (!MyHashMap1.remove("00"))
	{
		std::cout << "Error: remove bad return value." << std::endl;
		std::cout << "(see ~labcpp/www/ex1/HashSimpleCheck.cpp)" << std::endl;
		return 1;
	}

	if (MyHashMap2.size() != 2)
	{
		std::cout << "Error: Wrong size1, reported size:" << MyHashMap2.size() << std::endl;
		std::cout << "(see ~labcpp/www/ex1/HashSimpleCheck.cpp)" << std::endl;
		return 1;
	}
	if (MyHashMap1.size() != 1)
	{
		std::cout << "Error: Wrong size2." << std::endl;
		std::cout << "(see ~labcpp/www/ex1/HashSimpleCheck.cpp)" << std::endl;
		return 1;
	}
	if (!MyHashMap1.isIntersect(MyHashMap2) || !MyHashMap2.isIntersect(MyHashMap1))
	{
		std::cout << "Error: Wrong isIntersect1." << std::endl;
		std::cout << "(see ~labcpp/www/ex1/HashSimpleCheck.cpp)" << std::endl;
		return 1;
	}
	MyHashMap2.remove("0");
	if (MyHashMap1.isIntersect(MyHashMap2) || MyHashMap2.isIntersect(MyHashMap1))
	{
		std::cout << "Error: Wrong isIntersect2." << std::endl;
		std::cout << "(see ~labcpp/www/ex1/HashSimpleCheck.cpp)" << std::endl;
		return 1;
	}
	std::cout << "Pass HashSimpleCheck.cpp" << std::endl;
	return 0;
}
