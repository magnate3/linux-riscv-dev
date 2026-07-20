
/*
 * Copyright (C) Igor Sysoev
 * Copyright (C) Nginx, Inc.
 */


/*
14.2¹²ÏíÄÚ´æ
¹²ÏíÄÚ´æÊÇLinuxÏÂÌá¹©µÄ×î»ù±¾µÄ½ø³Ì¼äÍ¨ĞÅ·½·¨£¬ËüÍ¨¹ımmap»òÕßshmgetÏµÍ³µ÷ÓÃÔÚÄÚ´æÖĞ´´½¨ÁËÒ»¿éÁ¬ĞøµÄÏßĞÔµØÖ·¿Õ¼ä£¬¶øÍ¨¹ımunmap»ò
ÕßshmdtÏµÍ³µ÷ÓÃ¿ÉÒÔÊÍ·ÅÕâ¿éÄÚ´æ¡£Ê¹ÓÃ¹²ÏíÄÚ´æµÄºÃ´¦ÊÇµ±¶à¸ö½ø³ÌÊ¹ÓÃÍ¬Ò»¿é¹²ÏíÄÚ´æÊ±£¬ÔÚÈÎºÎÒ»¸ö½ø³ÌĞŞ¸ÄÁË¹²ÏíÄÚ´æÖĞµÄÄÚÈİºó£¬Æä
Ëû½ø³ÌÍ¨¹ı·ÃÎÊÕâ¶Î¹²ÏíÄÚ´æ¶¼ÄÜ¹»µÃµ½ĞŞ¸ÄºóµÄÄÚÈİ¡£
    ×¢Òâ:ËäÈ»mmap¿ÉÒÔÒÔ´ÅÅÌÎÄ¼şµÄ·½Ê½Ó³Éä¹²ÏíÄÚ´æ£¬µ«ÔÚNginx·â×°µÄ¹²ÏíÄÚ´æ²Ù×÷·½·¨ÖĞÊÇÃ»ÓĞÊ¹ÓÃµ½Ó³ÉäÎÄ¼ş¹¦ÄÜµÄ¡£

    ²Ù×÷ngx_shm_t½á¹¹ÌåµÄ·½·¨ÓĞÒÔÏÂÁ½¸ö£ºngx_shm_allocÓÃÓÚ·ÖÅäĞÂµÄ¹²ÏíÄÚ´æ£¬¶ø
ngx_shm_freeÓÃÓÚÊÍ·ÅÒÑ¾­´æÔÚµÄ¹²ÏíÄÚ´æ¡£ÔÚÃèÊöÕâÁ½¸ö·½·¨Ç°£¬ÏÈÒÔmmapÎªÀıËµÀÊ
LinuxÊÇÔõÑùÏòÓ¦ÓÃ³ÌĞòÌá¹©¹²ÏíÄÚ´æµÄ£¬ÈçÏÂËùÊ¾¡£
void *mmap (void *start, size_t length,  int prot,  int flags, int fd, off_t offset) ;
    mmap¿ÉÒÔ½«´ÅÅÌÎÄ¼şÓ³Éäµ½ÄÚ´æÖĞ£¬Ö±½Ó²Ù×÷ÄÚ´æÊ±LinuxÄÚºË½«¸ºÔğÍ¬²½ÄÚ´æºÍ´Å
ÅÌÎÄ¼şÖĞµÄÊı¾İ£¬fd²ÎÊı¾ÍÖ¸ÏòĞèÒªÍ¬²½µÄ´ÅÅÌÎÄ¼ş£¬¶øoffsetÔò´ú±í´ÓÎÄ¼şµÄÕâ¸öÆ«ÒÆÁ¿
´¦¿ªÊ¼¹²Ïí£¬µ±È»NginxÃ»ÓĞÊ¹ÓÃÕâÒ»ÌØĞÔ¡£µ±flags²ÎÊıÖĞ¼ÓÈëMAP ANON»òÕßMAP¡ª
ANONYMOUS²ÎÊıÊ±±íÊ¾²»Ê¹ÓÃÎÄ¼şÓ³Éä·½Ê½£¬ÕâÊ±fdºÍoffset²ÎÊı¾ÍÃ»ÓĞÒâÒå£¬Ò²²»
ĞèÒª´«µİÁË£¬´ËÊ±µÄmmap·½·¨ºÍngx_shm_allocµÄ¹¦ÄÜ¼¸ºõÍêÈ«ÏàÍ¬¡£length²ÎÊı¾ÍÊÇ½«
ÒªÔÚÄÚ´æÖĞ¿ª±ÙµÄÏßĞÔµØÖ·¿Õ¼ä´óĞ¡£¬¶øprot²ÎÊıÔòÊÇ²Ù×÷Õâ¶Î¹²ÏíÄÚ´æµÄ·½Ê½£¨ÈçÖ»¶Á
»òÕß¿É¶Á¿ÉĞ´£©£¬start²ÎÊıËµÃ÷Ï£ÍûµÄ¹²ÏíÄÚ´æÆğÊ¼Ó³ÉäµØÖ·£¬µ±È»£¬Í¨³£¶¼»á°ÑstartÉèÎª
NULL¿ÕÖ¸Õë¡£
    ÏÈÀ´¿´¿´ÈçºÎÊ¹ÓÃmmapÊµÏÖngx_shm_alloc·½·¨£¬´úÂëÈçÏÂ¡£
ngx_int_t ngx_shm_ alloc (ngx_shm_t  ~shm)
{
    £¯£¯¿ª±ÙÒ»¿éshm- >size´óĞ¡ÇÒ¿ÉÒÔ¶Á£¯Ğ´µÄ¹²ÏíÄÚ´æ£¬ÄÚ´æÊ×µØÖ·´æ·ÅÔÚaddrÖĞ
    shm->addr=  (uchar *)mmap (NULL,  shm->size,
    PROT_READ l PROT_WRITE,
    MAP_ANONIMAP_SHARED,  -1,o);
if (shm->addr == MAP_FAILED)
     return NGX ERROR;
}
    return NGX OK;
    )
    ÕâÀï²»ÔÙ½éÉÜshmget·½·¨ÉêÇë¹²ÏíÄÚ´æµÄ·½Ê½£¬ËüÓëÉÏÊö´úÂëÏàËÆ¡£
    µ±²»ÔÙÊ¹ÓÃ¹²ÏíÄÚ´æÊ±£¬ĞèÒªµ÷ÓÃmunmap»òÕßshmdtÀ´ÊÍ·Å¹²ÏíÄÚ´æ£¬ÕâÀï»¹ÊÇÒÔÓë
mmapÅä¶ÔµÄmunmapÎªÀıÀ´ËµÃ÷¡£
    ÆäÖĞ£¬start²ÎÊıÖ¸Ïò¹²ÏíÄÚ´æµÄÊ×µØÖ·£¬¶ølength²ÎÊı±íÊ¾Õâ¶Î¹²ÏíÄÚ´æµÄ³¤¶È¡£ÏÂÃæ
¿´¿´ngx_shm_free·½·¨ÊÇÔõÑùÍ¨¹ımunmapÀ´ÊÍ·Å¹²ÏíÄÚ´æµÄ¡£
    void  ngx_shm¡ªfree (ngx_shm_t¡ïshm)
    {
    £¯£¯Ê¹ÓÃngx_shm_tÖĞµÄaddrºÍsize²ÎÊıµ÷ÓÃmunmapÊÍ·Å¹²ÏíÄÚ´æ¼´¿É
    if  (munmap( (void¡ï)  shm->addr,  shm- >size)  ==~1)  (
    ngx_log_error (NGX¡ªLOG__  ALERT,  shm- >log,    ngx_errno,  ¡±munmap(%p,  %uz)
failed",  shm- >addr,   shm- >size)j
    )
    )
    Nginx¸÷½ø³Ì¼ä¹²ÏíÊı¾İµÄÖ÷Òª·½Ê½¾ÍÊÇÊ¹ÓÃ¹²ÏíÄÚ´æ£¨ÔÚÊ¹ÓÃ¹²ÏíÄÚ´æÊ±£¬Nginx -
°ãÊÇÓÉmaster½ø³Ì´´½¨£¬ÔÚmaster½ø³Ìfork³öworker×Ó½ø³Ìºó£¬ËùÓĞµÄ½ø³Ì¿ªÊ¼Ê¹ÓÃÕâ
¿éÄÚ´æÖĞµÄÊı¾İ£©¡£ÔÚ¿ª·¢NginxÄ£¿éÊ±Èç¹ûĞèÒªÊ¹ÓÃËü£¬²»·ÁÓÃNginxÒÑ¾­·â×°ºÃµÄngx_
shm¡ªalloc·½·¨ºÍngx_shm_free·½·¨£¬ËüÃÇÓĞ3ÖÖÊµÏÖ£¨²»Ó³ÉäÎÄ¼şÊ¹ÓÃmmap·ÖÅä¹²Ïí
ÄÚ´æ¡¢ÒÔ/dev/zeroÎÄ¼şÊ¹ÓÃmmapÓ³Éä¹²ÏíÄÚ´æ¡¢ÓÃshmgetµ÷ÓÃÀ´·ÖÅä¹²ÏíÄÚ´æ£©£¬¶ÔÓÚ
NginxµÄ¿çÆ½Ì¨ÌØĞÔ¿¼ÂÇµÃºÜÖÜµ½¡£ÏÂÃæÒÔÒ»¸öÍ³¼ÆHTTP¿ò¼ÜÁ¬½Ó×´¿öµÄÀı×ÓÀ´ËµÃ÷¹²Ïí
ÄÚ´æµÄÓÃ·¨¡£

*/

#include <ngx_config.h>
#include <ngx_core.h>

/*
ÔÚ¿ª·¢NginxÄ£¿éÊ±Èç¹ûĞèÒªÊ¹ÓÃËü£¬²»·ÁÓÃNginxÒÑ¾­·â×°ºÃµÄngx_shm_alloc·½·¨ºÍngx_shm_free·½·¨£¬ËüÃÇÓĞ3ÖÖÊµÏÖ£¨²»Ó³ÉäÎÄ¼şÊ¹ÓÃmmap·ÖÅä¹²Ïí
ÄÚ´æ¡¢ÒÔ/dev/zeroÎÄ¼şÊ¹ÓÃmmapÓ³Éä¹²ÏíÄÚ´æ¡¢ÓÃshmget(system-v±ê×¼)µ÷ÓÃÀ´·ÖÅä¹²ÏíÄÚ´æ£©
*/
//#ifÕâÀïµÄÈı¸ö¶¼defineÎª1£¬ËùÒÔÊ×ÏÈÂú×ãµÚÒ»¸öÌõ¼ş£¬Ñ¡ÔñµÚÒ»¸öifÖĞµÄ

/*
ÔÚ¿ª·¢NginxÄ£¿éÊ±Èç¹ûĞèÒªÊ¹ÓÃËü£¬²»·ÁÓÃNginxÒÑ¾­·â×°ºÃµÄngx_shm_alloc·½·¨ºÍngx_shm_free·½·¨£¬ËüÃÇÓĞ3ÖÖÊµÏÖ£¨²»Ó³ÉäÎÄ¼şÊ¹ÓÃmmap·ÖÅä¹²Ïí
ÄÚ´æ¡¢ÒÔ/dev/zeroÎÄ¼şÊ¹ÓÃmmapÓ³Éä¹²ÏíÄÚ´æ¡¢ÓÃshmget(system-v±ê×¼)µ÷ÓÃÀ´·ÖÅä¹²ÏíÄÚ´æ£©
*/

#include <sys/ipc.h>
#include <sys/shm.h>
//#define DEBUG_MULTIPLE_PROCS 2
void ngx_log_error(ngx_uint_t level, ngx_log_t *log,...){
   printf("errno==%d\n",errno);

}
#if (2== DEBUG_MULTIPLE_PROCS) 
#define SHARE_MEM_FILE "/var/run/test-ngx"
//static key_t shm_key=99888;
ngx_int_t
ngx_shm_alloc(ngx_shm_t *shm)
{
    int fd;
    unlink(SHARE_MEM_FILE);
    fd = open(SHARE_MEM_FILE, O_CREAT | O_RDWR, 0600);
    if (fd < 0){
    ngx_log_error(NGX_LOG_ALERT, shm->log, ngx_errno, "open file (%s) failed", SHARE_MEM_FILE);
    return NGX_ERROR;
    }
#if 1
    if (ftruncate(fd,  shm->size) < 0) {
	    goto err1;
    }
 #endif
    shm->addr = (u_char *) mmap(NULL, shm->size, PROT_READ|PROT_WRITE,  MAP_SHARED, fd, 0);
    if (shm->addr == MAP_FAILED) {
        ngx_log_error(NGX_LOG_ALERT, shm->log, ngx_errno, "mmap(MAP_ANON|MAP_SHARED, %uz) failed", shm->size);
	goto err1;
    }
    close(fd);
    return NGX_OK;
err1:
	            close(fd);
                    return NGX_ERROR;
}


void
ngx_shm_free(ngx_shm_t *shm)
{
    if (munmap((void *) shm->addr, shm->size) == -1) {
		            ngx_log_error(NGX_LOG_ALERT, shm->log, ngx_errno,
					                          "munmap(%p, %uz) failed", shm->addr, shm->size);
			        }
}
void
ngx_shm_destroy(ngx_shm_t *shm)
{
    ngx_shm_free(shm);
}
ngx_int_t
ngx_shm_attach(void * addr, ngx_int_t len,  int offset)
{
    int fd;
    fd = open(SHARE_MEM_FILE, O_RDWR, 0600);
    if (fd < 0){
    ngx_log_error(NGX_LOG_ALERT, NULL, ngx_errno, "open file (%s) failed", SHARE_MEM_FILE);
    return NGX_ERROR;
    }
    void * addr2 = mmap(addr, len, PROT_READ | PROT_WRITE, MAP_SHARED , fd, offset);
    if (addr2 == MAP_FAILED || addr != addr2)
       return -1;
    return NGX_OK;
}
#elif (1== DEBUG_MULTIPLE_PROCS) 
static key_t shm_key=99888;
ngx_int_t
ngx_shm_alloc(ngx_shm_t *shm)
{
    int  id;

    id = shmget(shm_key, 0, 0);
    if (id != -1)
    {
         shmctl(id, IPC_RMID, NULL);     //      ¿¿¿¿¿¿¿¿¿¿¿
    }
    id = shmget(shm_key, shm->size, (SHM_R|SHM_W|IPC_CREAT));

    if (id == -1) {
        ngx_log_error(NGX_LOG_ALERT, shm->log, ngx_errno,"shmget(%uz) failed", shm->size);
        return NGX_ERROR;
    }

    //ngx_log_debug1(NGX_LOG_DEBUG_CORE, shm->log, 0, "shmget id: %d", id);

    shm->addr = shmat(id, NULL, 0);

    if (shm->addr == (void *) -1) {
        ngx_log_error(NGX_LOG_ALERT, shm->log, ngx_errno, "shmat() failed");
    }

    if (shmctl(id, IPC_RMID, NULL) == -1) {
        ngx_log_error(NGX_LOG_ALERT, shm->log, ngx_errno,
                      "shmctl(IPC_RMID) failed");
    }

    return (shm->addr == (void *) -1) ? NGX_ERROR : NGX_OK;
}


void
ngx_shm_free(ngx_shm_t *shm)
{
    if (shmdt(shm->addr) == -1) {
        ngx_log_error(NGX_LOG_ALERT, shm->log, ngx_errno,
                      "shmdt(%p) failed", shm->addr);
    }
}
void
ngx_shm_destroy(ngx_shm_t *shm)
{
    int shmid = shmget(shm_key, 0, 0);
    if (shmid < 0)
            return;
   shmctl(shmid, IPC_RMID, NULL);
}
ngx_int_t
ngx_shm_attach()
{
    int shmid = shmget(shm_key, 0, 0);
    if (shmid < 0)
            return NGX_ERROR;
    if (shmat(shmid, NULL, 0) == NULL)
    {
        ngx_log_error(NGX_LOG_ALERT, NULL, ngx_errno,
                      "shmat(key %d) failed", shm_key);
            return NGX_ERROR;
    }
    return NGX_OK;
}
#else
ngx_int_t
ngx_shm_alloc(ngx_shm_t *shm)
{
    int  id;

    id = shmget(IPC_PRIVATE, shm->size, (SHM_R|SHM_W|IPC_CREAT));

    if (id == -1) {
        ngx_log_error(NGX_LOG_ALERT, shm->log, ngx_errno,"shmget(%uz) failed", shm->size);
        return NGX_ERROR;
    }

    //ngx_log_debug1(NGX_LOG_DEBUG_CORE, shm->log, 0, "shmget id: %d", id);

    shm->addr = shmat(id, NULL, 0);

    if (shm->addr == (void *) -1) {
        ngx_log_error(NGX_LOG_ALERT, shm->log, ngx_errno, "shmat() failed");
    }

    if (shmctl(id, IPC_RMID, NULL) == -1) {
        ngx_log_error(NGX_LOG_ALERT, shm->log, ngx_errno,
                      "shmctl(IPC_RMID) failed");
    }

    return (shm->addr == (void *) -1) ? NGX_ERROR : NGX_OK;
}


void
ngx_shm_free(ngx_shm_t *shm)
{
    if (shmdt(shm->addr) == -1) {
        ngx_log_error(NGX_LOG_ALERT, shm->log, ngx_errno,
                      "shmdt(%p) failed", shm->addr);
    }
}
#endif


