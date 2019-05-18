#pragma once
class smaller_than {
private:
		int tester;
public:

	bool operator()(int test) const { return test<tester; }

	smaller_than(int t);
	~smaller_than();
};

