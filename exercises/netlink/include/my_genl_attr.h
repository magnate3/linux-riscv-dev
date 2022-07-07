#ifndef _MY_GENL_ATTR_H_
#define _MY_GENL_ATTR_H_

enum my_genl_cmd {
	MY_CMD_UNSPEC,
	MY_CMD_ECHO,

	MY_CMD_END
};
#define MY_CMD_MAX (MY_CMD_END - 1)

enum my_genl_attrs {
	MY_ATTR_UNSPEC,
	MY_ATTR_MSG,

	MY_ATTR_END
};
#define MY_ATTR_MAX (MY_ATTR_END - 1)


#endif
