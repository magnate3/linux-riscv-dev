/* Copyright 2015 yiyuanzhong@gmail.com (Yiyuan Zhong)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __FULLNAT_H__
#define __FULLNAT_H__

extern int sysfs_initialize(void);
extern void sysfs_shutdown(void);

extern int procfs_initialize(void);
extern void procfs_shutdown(void);

extern int rip_initialize(void);
extern void rip_shutdown(void);

extern int g_mode_tcp6;
extern int g_mode_udp6;
extern int g_mode_tcp;
extern int g_mode_udp;

#endif /* __FULLNAT_H__ */
