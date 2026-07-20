#include "calendar_queue.h"
#include "calendar_queue_vector.h"

#include <stdlib.h>
#include <stdio.h>
#include <queue>
#include <chrono>

using namespace std;

struct Timer {
	Timer();
	~Timer();

	chrono::steady_clock::time_point begin;

	void reset();
	float since();
};

Timer::Timer() {
	begin = chrono::steady_clock::now();
}

Timer::~Timer() {
}

void Timer::reset() {
	begin = chrono::steady_clock::now();
}

float Timer::since() {
	return (float)(chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - begin).count())/1e6;
}

int main() {
	int seed = time(0);
	printf("seed %d\n", seed);
	srand(seed);
	calendar_queue<uint64_t> pq(20);
	calendar_queue_vector<uint64_t> cqv(20);
	std::priority_queue<uint64_t, std::vector<uint64_t>, std::greater<uint64_t> > cq;

	int init = 1000000;
	int m = 1000000;





	uint64_t now = std::numeric_limits<uint64_t>::max();
	for (int i = 0; i < init; i++) {
		uint64_t val = rand()%m;
		if (val < now) {
			now = val;
		}
		cq.push(val);
	}
	Timer t;
	for (int i = 0; i < 100000; i++) {
		int add = rand()%10;
		int sub = rand()%10;
		while (add-- >= 0) {
			uint64_t val = now+rand()%m;
			/*if (val < now) {
				now = val;
			}*/
			cq.push(rand()%m);
		}
		while (sub-- >= 0) {
			uint64_t val = cq.top();
			if (val > now) {
				now = val;
			}
			cq.pop();
		}
	}
	printf("priority_queue %f\n", t.since());
	while (not cq.empty()) {
		uint64_t val = cq.top();
		if (val > now) {
			now = val;
		}
		cq.pop();
	}





	for (int i = 0; i < init; i++) {
		pq.push(rand()%m);
	}
	t.reset();
	for (int i = 0; i < 100000; i++) {
		int add = rand()%10;
		int sub = rand()%10;
		while (add-- >= 0) {
			uint64_t val = pq.now+rand()%m;
			pq.push(val);
		}
		while (sub-- >= 0) {
			pq.pop();
		}
	}
	printf("calendar_queue %f\n", t.since());
	while (not pq.empty()) {
		pq.pop();
	}



	for (int i = 0; i < init; i++) {
		cqv.push(rand()%m);
	}
	t.reset();
	for (int i = 0; i < 100000; i++) {
		int add = rand()%10;
		int sub = rand()%10;
		while (add-- >= 0) {
			uint64_t val = cqv.now+rand()%m;
			cqv.push(val);
		}
		while (sub-- >= 0) {
			cqv.pop();
		}
	}
	printf("calendar_queue_vector %f\n", t.since());
	while (not cqv.empty()) {
		cqv.pop();
	}
}