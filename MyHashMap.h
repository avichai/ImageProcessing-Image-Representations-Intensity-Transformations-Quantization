/**
 * MyHashMap.h
 *
 * --------------------------------------------------------------------------------------
 * General: This class represents a Hashmap, a data structure that provide fast accession to
 *          data stored in.
 *          Depend on MyLinkedList
 *
 *  Methods: MyHashMap() - Constructor
 *  ~MyHashMap()         - Destructor
 *
 *  add                  - Add a string to the HashMap. Locate the entry of the relevant linked list in
 *                         the HashMap using the function myHashFunction, and add the element to
 *                         the end of that list.
 *                         If the element already exists in the HashMap, change its data to the
 *                         input parameter. Important: unlike our MyLinkedList, this HashMap will
 *                         contain at most one copy of each std::string.
 *
 *  remove               - Remove a string from the HashMap. Locate the entry of the relevant linked
 *                         list in the HashMap using the function myHashFunction, and remove the element
 *                         from it.
 *                         Return true on success, or false if the element wasn't in the HashMap.
 *
 *  isInHashMap            - Return true if the element is in the HashMap, or false otherwise.
 *                         If the element exists in the HashMap, return in 'data' its appropriate data
 *                         value. Otherwise don't change the value of 'data'.
 *
 *  size                 - Return number of elements stored in the HashMap.
 *
 *  isIntersect            - Return true if and only if there exists a string that belongs both to the
 *                         HashMap h1 and to this HashMap
 *
 *  totWeight            - Return the total weight of the hash elements
 *
 * --------------------------------------------------------------------------------------
 */
#ifndef MY_HASH_MAP_H
#define MY_HASH_MAP_H

#include <string>
#include "MyLinkedList.h"
#define HASH_SIZE 29	 // The number of entrances of the hash table
						 // should be a prim number

/**
 * The definition of the HashMap
 */
class MyHashMap
{
public:
	/**
	 * The hash function.
	 * Input parameter - any C++ string.
	 * Return value: the hash value - the sum of all the characters of the string
	 *   (according to their ASCII value) modulus HASH_SIZE. The hash value of the
	 *   empty string is 0 (since there are no characters, their sum according to
	 *   the ASCII value is 0).
	 * NOTE: In a better design the function would have belong to the string class
	 *	 (or to a class that is derived from std::string). We implement it as a "stand 
	 *	 alone" function since you didn't learn inheritance yet. Keep the function
	 *	 global (it's important to the automatic tests).
	 */
	static int myHashFunction(const std::string &str);

	/**
	 * Class constructor
	 */
	MyHashMap();

	/**
	 * Class destructor
	 */
	~MyHashMap();

	/**
	 * add a string to the HashMap. Locate the entry of the relevant linked list in
	 * the HashMap using the function myHashFunction, and add the element to the end of that list.
	 * If the element already exists in the HashMap, change its data to the
	 * input parameter. Important: unlike our MyLinkedList, that HashMap will contain at most
	 * one copy of each std::string.
	 */
	void add(const std::string &str, double data);
	
	/**
	 * remove a string from the HashMap. Locate the entry of the relevant linked list in
	 * the HashMap using the function myHashFunction, and remove the element from it.
	 * Return one on success, or zero if the element wasn't in the HashMap.
	 */
	size_t remove(const std::string &str);
	
	/**
	 * Return true if the element is in the HashMap, or false otherwise.
	 * If the element exists in the hash map, return in 'data' its appropriate data
	 * value. Otherwise don't change the value of 'data'.
	 */
	bool isInHashMap(const std::string &str, double &data) const;
	
	/**
	 * Return number of elements stored in the HashMap
	 */
	int size() const;
	
	/**
	 * Return true if and only if there exists a string that belong both to the
	 * HashMap h1 and to this HashMap
	 */
	bool isIntersect(const MyHashMap &h1) const;
	
	/**
	 * Return the total wight of the hash elements
	 */
	double totWeight() const;
private:
	MyHashMap(const MyHashMap&);
	MyHashMap& operator=(const MyHashMap&);
private:
	// The lists of the hash table
	MyLinkedList _hashT[HASH_SIZE];
	unsigned int _size;
};

#endif
