/**
 * Simple checks for the MyLinkedList implementation
 */
#include <iostream>
#include <string>
#include "MyLinkedList.h"
#define EPSILON 0.0001

static double myAbs(double d)
{
	if (d < 0)
	{
		return -d;
	}
	return d;
}

// Input: two doubles.
// Output: true if the two doubles are equal (different smaller than EPSILON), false otherwise.
static bool isDoublesEq(double d1, double d2)
{
	
	if (myAbs(d1-d2) <= EPSILON)
	{
		return true;
	}
	return false;

}

int main()
{

	MyLinkedList mylist;
	std::string firstWord = "aa";
	double firstVal = 1.5;
	std::string secondWord = "bb";
	double secVal = 2.2;
	std::string thirdWord = "ab";
	double thirdVal = 1.0;

	mylist.printList();
	std::cout << "Done print list" << std::endl << std::endl;

	mylist.add(firstWord, firstVal);
	mylist.add(secondWord, secVal);
	mylist.add(firstWord, thirdVal);
	mylist.add(thirdWord, firstVal);

	mylist.printList();
	std::cout << "Done print list" << std::endl << std::endl;

	double d;
	if (!mylist.isInList(thirdWord, d))
	{
		// isInList - check if thirdWord exists at least once in the list.
		// If it does - change 'd' so it will contain the data value of
		// one of thirdWord appearances and return 'true'. Otherwise
		// return 'false' and don't change the value of 'd'.
		std::cerr << "Error: Word " << thirdWord << " should exists in list." << std::endl;
		return 1;
	}
	else
	{
		if (!isDoublesEq(d, firstVal))
		{
			std::cerr << "Error: the appropriate value for the word " << thirdWord << " is " <<
						 firstVal << " while isInList returned " << d << std::endl;
			return 1;
		}
	}

	if (!mylist.remove(thirdWord))
	{
		// remove *all* appearances of thirdWord in the list (in our case one appearance)
		std::cerr << "Error: Removing " << thirdWord << " should have succeeded." << std::endl;
		return 1;
	}

	if (mylist.isInList(thirdWord, d))
	{
		std::cerr << "Error: Word " << thirdWord << " should not exists in list." << std::endl;
		return 1;
	}
	else
	{
		if (!isDoublesEq(d, firstVal))
		{
			std::cerr << "Error: value of 'd' shouldn't be changed by isInList for" << std::endl <<
						 "	   a word that doesn't exists in the list. " << std::endl;
			return 1;
		}
	}

	mylist.printList();
	std::cout << "Done print list" << std::endl << std::endl;

	std::cout << "Done test." << std::endl;
	return 0;
}

