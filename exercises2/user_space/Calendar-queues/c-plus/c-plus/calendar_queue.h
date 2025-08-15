#pragma once

#include <deque>
#include <vector>
#include <stdint.h>
#include <limits>

#include <stdio.h>

template <typename T>
struct default_priority {
	uint64_t operator()(const T &value) {
		return (uint64_t)value;
	}
};

template <typename T, typename P=default_priority<T> >
struct calendar_queue {
	struct event {
		event(size_t index) {
			next = nullptr;
			prev = nullptr;
			this->index = index;
		}

		~event() {
		}

		T value;
		size_t index;

		event *next;
		event *prev;
	};

	P priority;

	uint64_t count;
	uint64_t now;

	std::deque<event> events;
	event *unused;

	std::vector<std::pair<event*, event*> > calendar;

	// bit shift amounts
	int year;
	int day;
	int mindiff;

	calendar_queue(int year=14, int mindiff=4, P priority=P()) {
		this->count = 0;
		this->now = std::numeric_limits<uint64_t>::max();
		this->mindiff = mindiff;
		this->year = year;
		this->day = year < mindiff ? 0 : year-mindiff;
		this->priority = priority;
		this->unused = nullptr;
		calendar.resize(days(), std::pair<event*, event*>(nullptr, nullptr));
	}

	calendar_queue(const calendar_queue &q) {
		count = q.count;
		now = q.now;
		events = q.events;
		for (auto e = events.begin(); e != events.end(); e++) {
			if (e->next != nullptr) {
				e->next = &events[e->next->index];
			}
			if (e->prev != nullptr) {
				e->prev = &events[e->prev->index];
			}
		}
		unused = nullptr;
		if (q.unused != nullptr) {
			unused = &events[q.unused->index];
			unused->prev = nullptr;
		}
		event u = unused;
		for (event *e = q.unused->next; e != nullptr; e = e->next) {
			u->next = &events[e->index];
			u->next->prev = u;
			u = u->next;
		}
		u->next = nullptr;

		calendar = q.calendar;
		for (auto d = calendar.begin(); d != calendar.end(); d++) {
			if (d->first != nullptr) {
				d->first = &events[d->first->index];
			}
			if (d->second != nullptr) {
				d->second = &events[d->second->index];
			}
		}

		year = q.year;
		day = q.day;
		mindiff = q.mindiff;
	}

	~calendar_queue() {
	}

	uint64_t timeof(uint64_t day) {
		return day<<this->day;
	}

	uint64_t yearof(uint64_t time) {
		return time>>year;
	}

	uint64_t dayof(uint64_t time) {
		return (time>>day)&((1ul<<(year-day))-1ul);
	}

	uint64_t days() {
		return (1ul<<(year-day));
	}

	void shrink() {
		for (int i = 0; i < (int)calendar.size(); i+=2) {
			// merge calendar[i] and calendar[i+1]
			if (calendar[i].second == nullptr) {
				calendar[i] = calendar[i+1];
			} else if (calendar[i+1].first != nullptr) {
				event *e0 = calendar[i].second;
				event *e1 = calendar[i+1].second;
				uint64_t y0 = yearof(priority(e0->value));
				uint64_t y1 = yearof(priority(e1->value));
				uint64_t sy0 = yearof(priority(calendar[i].first->value));
				uint64_t sy1 = yearof(priority(calendar[i+1].first->value));
				while (e1 != nullptr) {
					if (y0 <= y1) {
						event *s1 = nullptr;
						if (y1 != sy1 and e0 != nullptr) {
							// if most events are in the same year, then we don't need to do
							// this search most of the time.
							// if e0 is nullptr, then we can move the entire list over
							for (s1 = e1->prev; s1 != nullptr and yearof(priority(s1->value)) == y1; s1 = s1->prev);
						}
						// by definition, e1 will not be nullptr because there would be
						// nothing to move over
						
						if (s1 == nullptr) {
							calendar[i+1].first->prev = e0;
						} /*else if (s1->next == nullptr) {
							// this shouldn't happen by definition of s1
						}*/ else if (s1->next != nullptr) {
							s1->next->prev = e0;
						}

						event *n = calendar[i].first;
						if (e0 == nullptr) {
							calendar[i].first->prev = e1;
						} else if (e0->next == nullptr) {
							calendar[i].second = e1;
						} else {
							e0->next->prev = e1;
						}

						/*if (e1->next == nullptr) {
							// already end of list, nothing needs to happen here
						} else */ if (e1->next != nullptr) {
							e1->next->prev = nullptr;
						}

						if (e0 == nullptr) {
							// this is broken
							e1->next = n;
							calendar[i].first = (s1 == nullptr ? calendar[i+1].first : s1->next);
						} else {
							e1->next = e0->next;
							e0->next = (s1 == nullptr ? calendar[i+1].first : s1->next);
						}

						if (s1 != nullptr) {
							s1->next = nullptr;
						}

						e1 = s1;
						if (e1 != nullptr) {
							y1 = yearof(priority(e1->value));
						}
					} else if (y0 != sy0) {
						for (e0 = e0->prev; e0 != nullptr and yearof(priority(e0->value)) == y0; e0 = e0->prev);
						y0 = yearof(priority(e0->value));
					} else {
						e0 = nullptr;
					}
				}
			}
			calendar[i+1].first = nullptr;
			calendar[i+1].second = nullptr;

			if (i != 0) {
				calendar[i/2] = calendar[i];
			}
		}
		day++;
		calendar.erase(calendar.begin()+days(), calendar.end());
	}

	void grow() {
		day--;
		calendar.resize(days(), std::pair<event*, event*>(nullptr, nullptr));
		for (int i = (int)calendar.size()-1; i >= 0; i--) {
			if (calendar[i].first == nullptr) {
				continue;
			}
			if (i != 0) {
				calendar[i*2] = calendar[i];
			}

			event *e = calendar[i*2].second;
			uint64_t y0 = yearof(priority(calendar[i*2].first->value));
			uint64_t t = priority(e->value);
			uint64_t y = yearof(t);
			uint64_t d = dayof(t);
			while (e != nullptr and (y > y0 or d != i*2)) {
				while (e != nullptr and d == i*2) {
					e = e->prev;
					if (e != nullptr) {
						t = priority(e->value);
						y = yearof(t);
						d = dayof(t);
						if (y == y0 and d == i*2) {
							e = nullptr;
						}
					}
				}
				if (e == nullptr) {
					break;
				}

				event *s = e;
				while (s != nullptr and dayof(priority(s->value)) == d) {
					s = s->prev;
				}

				if (e->next == nullptr) {
					calendar[i*2].second = s;
				} else {
					e->next->prev = s;
				}
				/*if (s == nullptr) {
					// Then calendar[i*2].first->prev is already nullptr
				} else*/ if (s != nullptr) {
					s->next->prev = nullptr;
				}

				event *n = e->next;
				e->next = calendar[i*2+1].first;
				
				if (s == nullptr) {
					calendar[i*2+1].first = calendar[i*2].first;
				} else {
					calendar[i*2+1].first = s->next;
				}

				if (e->next == nullptr) {
					calendar[i*2+1].second = e;
				} else {
					e->next->prev = e;
				}

				if (s == nullptr) {
					calendar[i*2].first = n;
				} else {
					s->next = n;
				}

				e = s;
				if (e != nullptr) {
					t = priority(e->value);
					y = yearof(t);
					d = dayof(t);
				}
			}

			if (i != 0) {
				calendar[i].first = nullptr;
				calendar[i].second = nullptr;
			}
		}
	}

	event *next(uint64_t time) {
		if (empty()) {
			return nullptr;
		}

		auto start = calendar.begin()+dayof(time);
		int y = yearof(time);
		event *m = nullptr;
		uint64_t mt;
		for (auto di = start; di != calendar.end(); di++) {
			for (event *e = di->first; e != nullptr; e = e->next) {
				uint64_t et = priority(e->value);
				if (et >= time) {
					if (yearof(et) == y) {
						return e;
					}

					if (m == nullptr or et < mt) {
						m = e;
						mt = et;
					}
					break;
				}
			}
		}
		y++;

		for (auto di = calendar.begin(); di != start; di++) {
			for (event *e = di->first; e != nullptr; e = e->next) {
				uint64_t et = priority(e->value);
				if (et >= time) {
					if (yearof(et) == y) {
						return e;
					}

					if (m == nullptr or et < mt) {
						m = e;
						mt = et;
					}
					break;
				}
			}
		}
		
		return m;
	}

	void add(event *e) {
		uint64_t t = priority(e->value);
		uint64_t d = dayof(t);

		event *n = calendar[d].first;
		while (n != nullptr and priority(n->value) < t) {
			n = n->next;
		}

		if (n == nullptr) {
			if (calendar[d].second == nullptr) {
				calendar[d].first = e;
				calendar[d].second = e;
			} else {
				calendar[d].second->next = e;
				e->prev = calendar[d].second;
				calendar[d].second = e;
			}
		} else {
			e->prev = n->prev;
			e->next = n;
			if (n->prev == nullptr) {
				calendar[d].first = e;
			} else {
				n->prev->next = e;
			}
			n->prev = e;
		}
		if (t < now) {
			now = t;
		}
		count++;
	}

	event *rem(event *e) {
		if (e == nullptr) {
			return nullptr;
		}

		uint64_t d = dayof(priority(e->value));
		if (e->prev == nullptr) {
			calendar[d].first = e->next;
		} else {
			e->prev->next = e->next;
		}

		if (e->next == nullptr) {
			calendar[d].second = e->prev;
		} else {
			e->next->prev = e->prev;
		}
		e->next = nullptr;
		e->prev = nullptr;
		count--;
		return e;
	}

	void set(event *e, T value) {
		if (priority(value) < priority(e->value)) {
			e->value = value;
			add(rem(e));
		}
	}

	event *push(T value) {
		event *result = nullptr;
		if (unused != nullptr) {
			result = unused;
			unused = unused->next;
			if (unused != nullptr) {
				unused->prev = nullptr;
			}
			result->next = nullptr;
		} else {
			events.push_back(event(events.size()));
			result = &events.back();
		}

		result->value = value;
		add(result);
		if (day > 0 and count > (days()<<1)) {
			grow();
		}
		return result;
	}

	T pop(event *e) {
		e = rem(e);
		if (e == nullptr) {
			return T();
		}
		e->next = unused;
		e->prev = nullptr;
		if (unused != nullptr) {
			unused->prev = e;
		}
		unused = e;
		if (year-day > mindiff and count < (days()>>1)) {
			shrink();
		}
		return e->value;
	}

	T pop(uint64_t time=std::numeric_limits<uint64_t>::max()) {
		if (time == std::numeric_limits<uint64_t>::max()) {
			time = now;
		}
		event *e = next(time);
		if (time == now and e != nullptr) {
			now = priority(e->value);
		}
		return pop(e);
	}

	uint64_t size() {
		return count;
	}

	bool empty() {
		return count == 0;
	}
};
