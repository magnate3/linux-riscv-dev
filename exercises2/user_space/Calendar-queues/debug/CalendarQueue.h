/*
 * A Calendar Queue implementation.
 *
 * Copyright (c) 2007, 2008 by Michael Neumann (mneumann@ntecs.de)
 *
 * Template parameters:
 *
 *   E: Element type
 *   ACC: Accessor
 *     priority -> real
 *     next -> E*&
 *   real: precision type
 * 
 */

#ifndef __YINSPIRE__CALENDAR_QUEUE__
#define __YINSPIRE__CALENDAR_QUEUE__

#include <assert.h>

template <typename E, class ACC = E, typename real=float>
class CalendarQueue
{
    typedef unsigned int I; // index type

  public:

    CalendarQueue(real year_width=1.0)
    {
      this->size_ = 0;
      this->year_width = year_width;
      this->day_width = year_width;

      this->current_year = 0;
      this->current_day = 0;

      this->num_days = 1;
      this->days = new E*[1];
      this->days[0] = NULL;
    }

    inline I
      size() const
      {
        return this->size_;
      }

    inline bool
      empty() const
      {
        return (this->size_ == 0);
      }

    void
      push(E *element)
      {
        const real priority = ACC::priority(element); 

        assert(priority >= 0.0);

        if (++this->size_ > 2*this->num_days) resize_double();

        // map priority to a day
        I day = ((I)(priority / this->day_width)) % this->num_days;

        // and sort element into that day
        insert_sorted(day, element);

        //
        // in case that the newly inserted element is smaller than the
        // currently smallest value we have to change the current year 
        // and day.
        //
        if (priority < min_start())
        {
          this->current_day  = day;
          this->current_year = (I)(priority / this->year_width);
        }
      }

    inline real min_start() const { return day_start() + year_start(); }
    inline real day_start() const { return this->current_day * this->day_width; }
    inline real year_start() const { return this->current_year * this->year_width; }
 
    E*
      pop()
      {
        assert(this->size_ > 0);

        if (--this->size_ < this->num_days/2 && this->num_days > 1) resize_half();

        real priority;
        real min_priority = INFINITY;
        real max_value_this_year = (this->current_year+1)*this->year_width;
        E *top;
        I i;

        for (i = this->current_day; i < this->num_days; i++)
        {
          top = this->days[i];
          if (top != NULL)
          {
            priority = ACC::priority(top);

            if (priority < max_value_this_year)
            {
              // remove top element
              this->days[i] = ACC::next(top); 
              this->current_day = i;
              ACC::next(top) = NULL;
              return top;
            }
            if (priority < min_priority) min_priority = priority;
          }
        }

        //
        // continue with the first element up to this->current_day
        //
        for (i = 0; i < this->current_day; i++)
        {
          top = this->days[i];
          if (top != NULL)
          {
            priority = ACC::priority(top);
            if (priority < min_priority) min_priority = priority;
          }
        }

        this->current_year = (I)(min_priority / this->year_width);
        this->current_day  = (I)(min_priority / this->day_width) % this->num_days;

        top = this->days[this->current_day];
        this->days[this->current_day] = ACC::next(top); 
        ACC::next(top) = NULL;

        return top;
      }

  protected:

    void
      insert_sorted(I day, E *element)
      {
        E *curr = this->days[day];
        E *prev = NULL; 
        const real priority = ACC::priority(element);

        //
        // Find the first element that is >= +element+.
        // Insert +element+ *before* that element.
        //
        while (curr != NULL && ACC::priority(curr) < priority)
        {
          prev = curr;
          curr = ACC::next(curr);
        }

        ACC::next(element) = curr;

        if (prev != NULL) ACC::next(prev) = element;
        else              this->days[day] = element;
      }

    /*
     * Each day is split into two days.
     */
    void
      resize_double()
      {
        E **new_days = new E*[2*this->num_days];

        const real dw = this->day_width / 2.0;
        E* c[2];

        for (I i = 0; i < this->num_days; i++)
        {
          c[0] = new_days[i*2]   = NULL;
          c[1] = new_days[i*2+1] = NULL;
          for (E *curr = this->days[i]; curr != NULL; curr = ACC::next(curr))
          {
            const I day = (I)(ACC::priority(curr) / dw) % 2;
            if (c[day] != NULL)
            {
              ACC::next(c[day]) = curr; 
            }
            else
            {
              new_days[i*2+day] = curr; 
            }
            c[day] = curr;
          }

          if (c[0] != NULL) ACC::next(c[0]) = NULL;
          if (c[1] != NULL) ACC::next(c[1]) = NULL;
        }

        delete [] this->days;

        this->days = new_days;
        this->num_days *= 2;
        this->day_width /= 2.0;
        this->current_year = 0;
        this->current_day = 0;
      }

    // TODO: merge
    void
      resize_half()
      {
        resize(this->num_days/2, this->day_width*2.0);
      }

    void
      resize(I new_num_days, real new_day_width)
      {
        E **old_days = this->days;
        this->days = new E*[new_num_days];
        E *element;

        // initialize this->days to NULL
        for (I i = 0; i < new_num_days; i++) this->days[i] = NULL;

        for (I i = 0; i < this->num_days; i++)
        {
          for (E *curr = old_days[i]; curr != NULL; )
          {
            const I day = ((I)(ACC::priority(curr) / new_day_width)) % new_num_days;
            element = curr; 
            curr = ACC::next(curr);
            insert_sorted(day, element);
          }
        }

        delete[] old_days;

        this->num_days = new_num_days;
        this->day_width = new_day_width;
        this->current_year = 0;
        this->current_day = 0;
      }

  private:

    I size_;

    I num_days;
    E** days;

    real day_width;
    real year_width;

    I current_year;
    I current_day;
};

#endif
