
#  net/ethtool

  ethtool -s eth0 speed 1000，即将eth0网口的速率设置为  

ethtool侧： do_sset --> ecmd.cmd = $\color{red}{ETHTOOL_SSET}$;  send_ioctl --> ioctl(ctx->fd, SIOCETHTOOL, &ctx->ifr);

linux侧：
 
 第一步：dev_ioctl根据   SIOCETHTOOL 命令字调用dev_ethtool
 ```
 
	case SIOCETHTOOL:
		dev_load(net, ifr.ifr_name);
		rtnl_lock();
		ret = dev_ethtool(net, &ifr);
		rtnl_unlock();
		if (!ret) {
			if (colon)
				*colon = ':';
			if (copy_to_user(arg, &ifr,
					 sizeof(struct ifreq)))
				ret = -EFAULT;
		}

 ```
 
 第二步：dev_ethtool 根据ETHTOOL_SSET 调用ethtool_set_settings
 
 ```
 case ETHTOOL_SSET:
		rc = ethtool_set_settings(dev, useraddr);
		break;
 ```
 第三步：ethtool_set_settings
 ```
 static int ethtool_set_settings(struct net_device *dev, void __user *useraddr)
{
	struct ethtool_cmd cmd;
 
	ASSERT_RTNL();
 
	if (copy_from_user(&cmd, useraddr, sizeof(cmd)))
		return -EFAULT;
 
	/* first, try new %ethtool_link_ksettings API. */
	if (dev->ethtool_ops->set_link_ksettings) {
		struct ethtool_link_ksettings link_ksettings;
 
		if (!convert_legacy_settings_to_link_ksettings(&link_ksettings,
							       &cmd))
			return -EINVAL;
 
		link_ksettings.base.cmd = ETHTOOL_SLINKSETTINGS;
		link_ksettings.base.link_mode_masks_nwords
			= __ETHTOOL_LINK_MODE_MASK_NU32;
		return dev->ethtool_ops->set_link_ksettings(dev,
							    &link_ksettings);
	}
 
	/* legacy %ethtool_cmd API */
 
	/* TODO: return -EOPNOTSUPP when ethtool_ops::get_settings
	 * disappears internally
	 */
 
	if (!dev->ethtool_ops->set_settings)
		return -EOPNOTSUPP;
 
	return dev->ethtool_ops->set_settings(dev, &cmd);
}
 ```
 
## macb_set_link_ksettings

```
static const struct ethtool_ops macb_ethtool_ops = {
        .get_regs_len           = macb_get_regs_len,
        .get_regs               = macb_get_regs,
        .get_link               = ethtool_op_get_link,
        .get_ts_info            = ethtool_op_get_ts_info,
        .get_wol                = macb_get_wol,
        .set_wol                = macb_set_wol,
        .get_link_ksettings     = macb_get_link_ksettings,
        .set_link_ksettings     = macb_set_link_ksettings,
        .get_ringparam          = macb_get_ringparam,
        .set_ringparam          = macb_set_ringparam,
};
```
 
```
static int macb_set_link_ksettings(struct net_device *netdev,
                                   const struct ethtool_link_ksettings *kset)
{
        struct macb *bp = netdev_priv(netdev);

        return phylink_ethtool_ksettings_set(bp->phylink, kset);
}
 ```