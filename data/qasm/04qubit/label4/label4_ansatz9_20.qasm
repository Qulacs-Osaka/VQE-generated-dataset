OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(1.0388907740399222) q[0];
ry(1.2221676607507304) q[1];
cx q[0],q[1];
ry(2.637089898162296) q[0];
ry(-3.0615009315177746) q[1];
cx q[0],q[1];
ry(2.512018435434094) q[2];
ry(3.0483170289356236) q[3];
cx q[2],q[3];
ry(1.430052077830748) q[2];
ry(-2.6496906168641328) q[3];
cx q[2],q[3];
ry(2.2137787413963252) q[0];
ry(-2.3914102969071696) q[2];
cx q[0],q[2];
ry(-0.4742656686505162) q[0];
ry(2.914296963294757) q[2];
cx q[0],q[2];
ry(-2.6421704199778007) q[1];
ry(1.881675345080328) q[3];
cx q[1],q[3];
ry(0.5892850824550298) q[1];
ry(-2.273778367576746) q[3];
cx q[1],q[3];
ry(0.9855393811958093) q[0];
ry(-2.3700835078151563) q[3];
cx q[0],q[3];
ry(1.7373180755404314) q[0];
ry(0.7021598138955144) q[3];
cx q[0],q[3];
ry(0.4015011980067876) q[1];
ry(2.3046334129976054) q[2];
cx q[1],q[2];
ry(-1.7154173574918579) q[1];
ry(2.8711128309586615) q[2];
cx q[1],q[2];
ry(0.4927085689876316) q[0];
ry(2.5436693639035206) q[1];
cx q[0],q[1];
ry(-0.6825920213023873) q[0];
ry(-2.0185983692674316) q[1];
cx q[0],q[1];
ry(-2.1394510739193993) q[2];
ry(-0.026965752205999505) q[3];
cx q[2],q[3];
ry(-2.5468106011581595) q[2];
ry(-0.5730224879759119) q[3];
cx q[2],q[3];
ry(0.18215952248097966) q[0];
ry(0.23874161176418074) q[2];
cx q[0],q[2];
ry(0.5620590226761388) q[0];
ry(1.2027951160054773) q[2];
cx q[0],q[2];
ry(0.8764502893945575) q[1];
ry(2.482868234678786) q[3];
cx q[1],q[3];
ry(-3.1066495321864833) q[1];
ry(1.24719724245762) q[3];
cx q[1],q[3];
ry(2.2301862112743676) q[0];
ry(1.490234387196515) q[3];
cx q[0],q[3];
ry(-0.6715862580325548) q[0];
ry(1.1351119843728885) q[3];
cx q[0],q[3];
ry(0.7707713201088184) q[1];
ry(0.5347402514260127) q[2];
cx q[1],q[2];
ry(-1.124560689927275) q[1];
ry(2.5384134215931375) q[2];
cx q[1],q[2];
ry(-2.725956489717543) q[0];
ry(-1.9354014287786532) q[1];
cx q[0],q[1];
ry(0.3711071636394442) q[0];
ry(-2.626942815246844) q[1];
cx q[0],q[1];
ry(0.23392609926478694) q[2];
ry(-2.5157613444899285) q[3];
cx q[2],q[3];
ry(-1.611233445407061) q[2];
ry(-2.3735699968796378) q[3];
cx q[2],q[3];
ry(-2.119124914914957) q[0];
ry(-1.8191597296798445) q[2];
cx q[0],q[2];
ry(2.8308938475324994) q[0];
ry(0.49544808423932) q[2];
cx q[0],q[2];
ry(1.4612873633755878) q[1];
ry(-0.04808390055194658) q[3];
cx q[1],q[3];
ry(0.8093715284958467) q[1];
ry(-2.022105714854077) q[3];
cx q[1],q[3];
ry(2.2318111679512587) q[0];
ry(-1.5440031147857551) q[3];
cx q[0],q[3];
ry(1.6956273223070735) q[0];
ry(2.018977357616347) q[3];
cx q[0],q[3];
ry(-0.5842737586143922) q[1];
ry(1.0655076118498084) q[2];
cx q[1],q[2];
ry(-2.24555525625205) q[1];
ry(-1.8072307402480545) q[2];
cx q[1],q[2];
ry(2.8307241780294214) q[0];
ry(2.018154079653245) q[1];
cx q[0],q[1];
ry(0.6265546549257065) q[0];
ry(3.0427485802080696) q[1];
cx q[0],q[1];
ry(-2.718535901417085) q[2];
ry(-2.3571633654962905) q[3];
cx q[2],q[3];
ry(2.956453907040046) q[2];
ry(1.4951775621219463) q[3];
cx q[2],q[3];
ry(0.33834592882503073) q[0];
ry(1.7360461644593905) q[2];
cx q[0],q[2];
ry(2.6052807563231046) q[0];
ry(-0.8766295506887003) q[2];
cx q[0],q[2];
ry(-2.0262635551255226) q[1];
ry(-1.1658467004987294) q[3];
cx q[1],q[3];
ry(-1.051938181763167) q[1];
ry(1.3715271966123952) q[3];
cx q[1],q[3];
ry(2.9974222672091715) q[0];
ry(-1.0395468713022444) q[3];
cx q[0],q[3];
ry(0.27401433155750166) q[0];
ry(-0.19251453810423708) q[3];
cx q[0],q[3];
ry(2.5308898322492417) q[1];
ry(-2.75537887475225) q[2];
cx q[1],q[2];
ry(0.15136963276212345) q[1];
ry(-0.33942233238134845) q[2];
cx q[1],q[2];
ry(1.0266229693005133) q[0];
ry(-0.7339492323098109) q[1];
cx q[0],q[1];
ry(-2.5538865009231055) q[0];
ry(1.5320056065616203) q[1];
cx q[0],q[1];
ry(1.1895570458409566) q[2];
ry(0.14435264744299126) q[3];
cx q[2],q[3];
ry(-1.668458424650722) q[2];
ry(1.4121183790219043) q[3];
cx q[2],q[3];
ry(-1.9451366390498652) q[0];
ry(-2.217777111858017) q[2];
cx q[0],q[2];
ry(3.1125771731685847) q[0];
ry(2.5611239023973638) q[2];
cx q[0],q[2];
ry(0.01364737677634764) q[1];
ry(2.5779068561147365) q[3];
cx q[1],q[3];
ry(0.9508978107452508) q[1];
ry(-1.3467707190316853) q[3];
cx q[1],q[3];
ry(-2.9023050565897837) q[0];
ry(1.4106780630222104) q[3];
cx q[0],q[3];
ry(-0.6148695043134265) q[0];
ry(1.0800763133777265) q[3];
cx q[0],q[3];
ry(-1.5090097213920517) q[1];
ry(1.6495930084002906) q[2];
cx q[1],q[2];
ry(1.8918199651892198) q[1];
ry(-1.354366148626558) q[2];
cx q[1],q[2];
ry(0.5752727014399135) q[0];
ry(-2.127499492903959) q[1];
cx q[0],q[1];
ry(0.6172180161295214) q[0];
ry(-0.16797755495864838) q[1];
cx q[0],q[1];
ry(-1.8168679320792096) q[2];
ry(-2.902088834636381) q[3];
cx q[2],q[3];
ry(-1.9922456645331077) q[2];
ry(1.573074327169322) q[3];
cx q[2],q[3];
ry(0.3455097446872732) q[0];
ry(-0.6187068069452906) q[2];
cx q[0],q[2];
ry(0.5495391417531749) q[0];
ry(-1.968652385371251) q[2];
cx q[0],q[2];
ry(2.3472236784045846) q[1];
ry(1.9163024819573353) q[3];
cx q[1],q[3];
ry(0.5056805144390514) q[1];
ry(-0.22418730543422605) q[3];
cx q[1],q[3];
ry(2.2158606291303444) q[0];
ry(1.1002543534588876) q[3];
cx q[0],q[3];
ry(-2.8407334484047224) q[0];
ry(-2.757767896233186) q[3];
cx q[0],q[3];
ry(-0.6999080651609944) q[1];
ry(1.3650238315818177) q[2];
cx q[1],q[2];
ry(-1.985090077397946) q[1];
ry(-1.196913896000237) q[2];
cx q[1],q[2];
ry(2.4955961503815853) q[0];
ry(3.1297780658630825) q[1];
cx q[0],q[1];
ry(-1.6479582831928594) q[0];
ry(-0.43513806356831675) q[1];
cx q[0],q[1];
ry(-1.1906844953540108) q[2];
ry(2.1190596913726694) q[3];
cx q[2],q[3];
ry(1.0373091209974161) q[2];
ry(-1.8636604711179485) q[3];
cx q[2],q[3];
ry(-0.03193338552433361) q[0];
ry(-2.4977801302350637) q[2];
cx q[0],q[2];
ry(-0.4642619603867952) q[0];
ry(-1.5989175217034306) q[2];
cx q[0],q[2];
ry(-0.8701584351169567) q[1];
ry(-0.6644005280514413) q[3];
cx q[1],q[3];
ry(1.2538431984261122) q[1];
ry(1.5788084127547537) q[3];
cx q[1],q[3];
ry(-2.7283130860832823) q[0];
ry(1.3385622588669672) q[3];
cx q[0],q[3];
ry(0.3910190848448618) q[0];
ry(-0.21139718803335303) q[3];
cx q[0],q[3];
ry(3.0384345848894627) q[1];
ry(-2.6846228459672528) q[2];
cx q[1],q[2];
ry(1.6047227899438468) q[1];
ry(1.7394665421015567) q[2];
cx q[1],q[2];
ry(1.9132195005985049) q[0];
ry(1.1897024913705248) q[1];
cx q[0],q[1];
ry(-2.147686984681285) q[0];
ry(-0.8236550792438416) q[1];
cx q[0],q[1];
ry(2.574765277326951) q[2];
ry(-2.04569221400717) q[3];
cx q[2],q[3];
ry(0.8385876069091136) q[2];
ry(-1.818310873287664) q[3];
cx q[2],q[3];
ry(-1.1920239516851714) q[0];
ry(1.8105748474349117) q[2];
cx q[0],q[2];
ry(2.331191179168765) q[0];
ry(0.8331278262984947) q[2];
cx q[0],q[2];
ry(2.004805971850367) q[1];
ry(-0.4736767489595257) q[3];
cx q[1],q[3];
ry(2.763356937361911) q[1];
ry(0.7682174476990058) q[3];
cx q[1],q[3];
ry(-0.1391654277436185) q[0];
ry(0.5558224875500539) q[3];
cx q[0],q[3];
ry(2.444809046210101) q[0];
ry(2.37063478695822) q[3];
cx q[0],q[3];
ry(-2.7351634644122877) q[1];
ry(-2.612579937336788) q[2];
cx q[1],q[2];
ry(1.0849995928072784) q[1];
ry(-2.5478139786338017) q[2];
cx q[1],q[2];
ry(-0.038713113106261604) q[0];
ry(-1.0686969008365912) q[1];
cx q[0],q[1];
ry(2.838181809373497) q[0];
ry(0.45433070989561397) q[1];
cx q[0],q[1];
ry(-0.6167561863835472) q[2];
ry(-1.2206128095382356) q[3];
cx q[2],q[3];
ry(0.25002122200579424) q[2];
ry(-2.998287480520545) q[3];
cx q[2],q[3];
ry(2.1447867272253247) q[0];
ry(-0.12555844946690708) q[2];
cx q[0],q[2];
ry(2.465580082432761) q[0];
ry(-0.6343607391225179) q[2];
cx q[0],q[2];
ry(1.3084492448541134) q[1];
ry(-2.8565039685561366) q[3];
cx q[1],q[3];
ry(1.3495974955535166) q[1];
ry(-2.9188498061227324) q[3];
cx q[1],q[3];
ry(-2.2568218512852054) q[0];
ry(-2.4239028248238443) q[3];
cx q[0],q[3];
ry(-0.28308972038067454) q[0];
ry(-2.758034705839659) q[3];
cx q[0],q[3];
ry(-1.997276049085114) q[1];
ry(0.6029162005189637) q[2];
cx q[1],q[2];
ry(-1.648425952757828) q[1];
ry(-2.378858919337797) q[2];
cx q[1],q[2];
ry(2.257296760579906) q[0];
ry(0.9357374331089847) q[1];
cx q[0],q[1];
ry(1.142584724078624) q[0];
ry(-1.9545913754304374) q[1];
cx q[0],q[1];
ry(2.158244753824338) q[2];
ry(2.2575496339691923) q[3];
cx q[2],q[3];
ry(-0.7391126258316499) q[2];
ry(-2.505266277118637) q[3];
cx q[2],q[3];
ry(2.962784349045788) q[0];
ry(-1.0135483518886375) q[2];
cx q[0],q[2];
ry(2.772342999735333) q[0];
ry(-0.18140750786091003) q[2];
cx q[0],q[2];
ry(0.7037883339618096) q[1];
ry(-1.6381361491579876) q[3];
cx q[1],q[3];
ry(0.0943708814476492) q[1];
ry(-0.707386091964092) q[3];
cx q[1],q[3];
ry(1.4876552520724609) q[0];
ry(-1.1647984711725021) q[3];
cx q[0],q[3];
ry(0.14928471374360797) q[0];
ry(0.1806857152036234) q[3];
cx q[0],q[3];
ry(0.2325403400934881) q[1];
ry(-1.0713203593694889) q[2];
cx q[1],q[2];
ry(-2.3961606114623435) q[1];
ry(-1.6380435577946875) q[2];
cx q[1],q[2];
ry(2.4027616434898436) q[0];
ry(0.43705313311148775) q[1];
cx q[0],q[1];
ry(1.6459596137874772) q[0];
ry(0.31088651057588973) q[1];
cx q[0],q[1];
ry(2.8117780006906403) q[2];
ry(0.42771669053125677) q[3];
cx q[2],q[3];
ry(1.8992403775819646) q[2];
ry(2.954891998407527) q[3];
cx q[2],q[3];
ry(-1.938854759494835) q[0];
ry(-2.495664695733659) q[2];
cx q[0],q[2];
ry(2.6411897102730304) q[0];
ry(1.7078545831675285) q[2];
cx q[0],q[2];
ry(-0.6712623842853551) q[1];
ry(-1.2955043962738861) q[3];
cx q[1],q[3];
ry(2.776000965727452) q[1];
ry(-2.580328367474017) q[3];
cx q[1],q[3];
ry(-0.8383402793319743) q[0];
ry(0.6566756485696111) q[3];
cx q[0],q[3];
ry(-0.8269637281418588) q[0];
ry(2.02945295036269) q[3];
cx q[0],q[3];
ry(2.609096793900921) q[1];
ry(0.5989203032630606) q[2];
cx q[1],q[2];
ry(2.7641131894265705) q[1];
ry(0.9076049893719035) q[2];
cx q[1],q[2];
ry(2.5757477962798836) q[0];
ry(1.3689996370613675) q[1];
cx q[0],q[1];
ry(0.31883426681723304) q[0];
ry(1.6183171553679223) q[1];
cx q[0],q[1];
ry(-1.93531492639317) q[2];
ry(0.5502227801125998) q[3];
cx q[2],q[3];
ry(-1.3443898464683708) q[2];
ry(-1.8875533645624705) q[3];
cx q[2],q[3];
ry(-1.2934979787358885) q[0];
ry(-2.846905261125584) q[2];
cx q[0],q[2];
ry(-0.08263211990551156) q[0];
ry(2.0773338532992924) q[2];
cx q[0],q[2];
ry(-0.5204290663493678) q[1];
ry(-0.9284141793770143) q[3];
cx q[1],q[3];
ry(-1.6815821364461085) q[1];
ry(-0.18550794118576341) q[3];
cx q[1],q[3];
ry(-0.21664407203331937) q[0];
ry(-0.5983293003267812) q[3];
cx q[0],q[3];
ry(-2.18407417218776) q[0];
ry(1.5331674968020828) q[3];
cx q[0],q[3];
ry(0.38323754534886945) q[1];
ry(-0.17125472495568506) q[2];
cx q[1],q[2];
ry(-1.1999775291477617) q[1];
ry(-1.9667595977421513) q[2];
cx q[1],q[2];
ry(-2.172088134653605) q[0];
ry(-2.550969971000685) q[1];
cx q[0],q[1];
ry(1.9881462273013093) q[0];
ry(2.6690885278694134) q[1];
cx q[0],q[1];
ry(1.7561998157983432) q[2];
ry(-0.07815720074001348) q[3];
cx q[2],q[3];
ry(-1.6713972699520379) q[2];
ry(0.8024009719427951) q[3];
cx q[2],q[3];
ry(-1.4902231147991405) q[0];
ry(-1.0059908579159405) q[2];
cx q[0],q[2];
ry(-1.4333493500279033) q[0];
ry(2.0673511982481454) q[2];
cx q[0],q[2];
ry(2.6419462373452323) q[1];
ry(-2.8063285139586904) q[3];
cx q[1],q[3];
ry(1.3079651909925818) q[1];
ry(-1.5222665308783212) q[3];
cx q[1],q[3];
ry(-2.7812089895677325) q[0];
ry(3.018611885586937) q[3];
cx q[0],q[3];
ry(0.0692544418396787) q[0];
ry(-0.5406748221232537) q[3];
cx q[0],q[3];
ry(-1.6961821042487648) q[1];
ry(0.3309626619096449) q[2];
cx q[1],q[2];
ry(-2.730849488650891) q[1];
ry(-1.3310767986066538) q[2];
cx q[1],q[2];
ry(-1.6233222742791862) q[0];
ry(-1.44973716840823) q[1];
cx q[0],q[1];
ry(0.26228712340401295) q[0];
ry(-1.1851085432075603) q[1];
cx q[0],q[1];
ry(2.9386392413884637) q[2];
ry(-0.6809440533858164) q[3];
cx q[2],q[3];
ry(-0.8176521251975126) q[2];
ry(-2.866043104147203) q[3];
cx q[2],q[3];
ry(0.8225571743144355) q[0];
ry(2.8092629758832315) q[2];
cx q[0],q[2];
ry(0.014184557675884337) q[0];
ry(1.361710838188096) q[2];
cx q[0],q[2];
ry(1.1386379212945252) q[1];
ry(2.3099246112966507) q[3];
cx q[1],q[3];
ry(0.1737948910391604) q[1];
ry(-2.824304627198229) q[3];
cx q[1],q[3];
ry(-1.6326531734223684) q[0];
ry(-2.662111069848769) q[3];
cx q[0],q[3];
ry(-0.41214359684916074) q[0];
ry(0.763957694066554) q[3];
cx q[0],q[3];
ry(0.21349127588830896) q[1];
ry(3.006539390343613) q[2];
cx q[1],q[2];
ry(-2.9760690127461857) q[1];
ry(-1.3483137304012391) q[2];
cx q[1],q[2];
ry(1.3442324066845002) q[0];
ry(2.3026405100100753) q[1];
cx q[0],q[1];
ry(2.7363453478750737) q[0];
ry(-1.99647822031127) q[1];
cx q[0],q[1];
ry(2.1523582399980206) q[2];
ry(-1.6141564810453004) q[3];
cx q[2],q[3];
ry(-0.17135974211163418) q[2];
ry(2.2767741724064683) q[3];
cx q[2],q[3];
ry(1.4108023162945527) q[0];
ry(-0.24835928918201172) q[2];
cx q[0],q[2];
ry(-1.5354949776094813) q[0];
ry(-2.7410973148271602) q[2];
cx q[0],q[2];
ry(-0.362884660938934) q[1];
ry(2.4134630589295387) q[3];
cx q[1],q[3];
ry(2.660534435923336) q[1];
ry(-2.073515840708602) q[3];
cx q[1],q[3];
ry(1.163734041447828) q[0];
ry(2.239180629715292) q[3];
cx q[0],q[3];
ry(2.9913617370284054) q[0];
ry(-0.22348742444759043) q[3];
cx q[0],q[3];
ry(-0.7497877964232919) q[1];
ry(0.43558378561336486) q[2];
cx q[1],q[2];
ry(1.006692871805397) q[1];
ry(2.790256954209005) q[2];
cx q[1],q[2];
ry(0.3510922740082414) q[0];
ry(-2.9729999980641817) q[1];
cx q[0],q[1];
ry(-0.2509246022405094) q[0];
ry(1.7522247786366423) q[1];
cx q[0],q[1];
ry(1.5311167332059679) q[2];
ry(-2.4174993758080427) q[3];
cx q[2],q[3];
ry(-1.9861179479229758) q[2];
ry(-0.36676518362746896) q[3];
cx q[2],q[3];
ry(-2.6942253784251635) q[0];
ry(2.2824231295257547) q[2];
cx q[0],q[2];
ry(0.06216863396260757) q[0];
ry(2.467290413405364) q[2];
cx q[0],q[2];
ry(3.090866366394733) q[1];
ry(2.9912300490228283) q[3];
cx q[1],q[3];
ry(2.7613438202573293) q[1];
ry(-1.1684128497527646) q[3];
cx q[1],q[3];
ry(-3.1307526116290214) q[0];
ry(-1.8046109391835676) q[3];
cx q[0],q[3];
ry(1.3184957758491687) q[0];
ry(1.0737468646330584) q[3];
cx q[0],q[3];
ry(-1.6474071575202889) q[1];
ry(-2.068585720477196) q[2];
cx q[1],q[2];
ry(-2.3056318888555682) q[1];
ry(3.0561840556244335) q[2];
cx q[1],q[2];
ry(-0.8807814191667367) q[0];
ry(-0.36821264758793565) q[1];
cx q[0],q[1];
ry(-2.7519515908724945) q[0];
ry(-2.9859648001428827) q[1];
cx q[0],q[1];
ry(1.1297638127362135) q[2];
ry(0.8771591313173187) q[3];
cx q[2],q[3];
ry(-1.4650777153963361) q[2];
ry(0.9375796665806329) q[3];
cx q[2],q[3];
ry(-2.2105323593933726) q[0];
ry(-1.9212735457320649) q[2];
cx q[0],q[2];
ry(-1.2668559195801885) q[0];
ry(0.36697090513305036) q[2];
cx q[0],q[2];
ry(-1.0514956945593612) q[1];
ry(-3.0804579314763525) q[3];
cx q[1],q[3];
ry(-1.611630260064544) q[1];
ry(1.5015646474621656) q[3];
cx q[1],q[3];
ry(0.619175333632648) q[0];
ry(-0.07389929437820049) q[3];
cx q[0],q[3];
ry(1.5440783086001302) q[0];
ry(1.009231651418304) q[3];
cx q[0],q[3];
ry(-0.8169875027010525) q[1];
ry(2.277240249569079) q[2];
cx q[1],q[2];
ry(-0.45348388415792007) q[1];
ry(-0.11035571051469954) q[2];
cx q[1],q[2];
ry(-1.153634151312823) q[0];
ry(0.10900370690794166) q[1];
cx q[0],q[1];
ry(0.314233100734957) q[0];
ry(-0.38116991285586177) q[1];
cx q[0],q[1];
ry(1.9353398078144979) q[2];
ry(-2.4314631370161894) q[3];
cx q[2],q[3];
ry(-1.3800053765338556) q[2];
ry(-0.6724893324375323) q[3];
cx q[2],q[3];
ry(-2.6869660774202866) q[0];
ry(0.980659702903055) q[2];
cx q[0],q[2];
ry(1.298045895399782) q[0];
ry(1.5353684707492603) q[2];
cx q[0],q[2];
ry(1.206277989728732) q[1];
ry(1.0268891662256747) q[3];
cx q[1],q[3];
ry(-2.0424734638646007) q[1];
ry(0.6835962507835717) q[3];
cx q[1],q[3];
ry(0.7611224453443418) q[0];
ry(-0.4630861188972464) q[3];
cx q[0],q[3];
ry(-0.6690334388304275) q[0];
ry(1.5838877656962211) q[3];
cx q[0],q[3];
ry(-2.1422946947903627) q[1];
ry(-1.9534313914859314) q[2];
cx q[1],q[2];
ry(1.1362310174310402) q[1];
ry(0.7354160839818823) q[2];
cx q[1],q[2];
ry(-1.714328649073587) q[0];
ry(0.13179408498019815) q[1];
cx q[0],q[1];
ry(2.0728247016175785) q[0];
ry(2.3642734426652736) q[1];
cx q[0],q[1];
ry(-1.4908062731409268) q[2];
ry(0.6184832765508068) q[3];
cx q[2],q[3];
ry(-2.0735140813289448) q[2];
ry(1.6451918867041133) q[3];
cx q[2],q[3];
ry(-1.103723764139537) q[0];
ry(3.0503477142203024) q[2];
cx q[0],q[2];
ry(-0.5087186441869901) q[0];
ry(-2.265514587943237) q[2];
cx q[0],q[2];
ry(-1.412732517939852) q[1];
ry(0.6354412595651588) q[3];
cx q[1],q[3];
ry(2.657649799355853) q[1];
ry(-0.31483758597966655) q[3];
cx q[1],q[3];
ry(-1.6445284576450279) q[0];
ry(-1.0224289302085738) q[3];
cx q[0],q[3];
ry(-2.5491816609035536) q[0];
ry(-1.833113328337383) q[3];
cx q[0],q[3];
ry(2.1026579830802574) q[1];
ry(-2.1273563946336402) q[2];
cx q[1],q[2];
ry(-2.0832402247816715) q[1];
ry(0.834477218103543) q[2];
cx q[1],q[2];
ry(0.6137919781745538) q[0];
ry(2.0763697641994456) q[1];
cx q[0],q[1];
ry(2.6749148334381787) q[0];
ry(1.8512131798813414) q[1];
cx q[0],q[1];
ry(0.4406627464132524) q[2];
ry(1.1191912031342717) q[3];
cx q[2],q[3];
ry(2.8914774504850946) q[2];
ry(-1.5562072881926454) q[3];
cx q[2],q[3];
ry(0.1611387020865288) q[0];
ry(-2.111002528003193) q[2];
cx q[0],q[2];
ry(-3.0538912192601644) q[0];
ry(1.7647041770995051) q[2];
cx q[0],q[2];
ry(-0.589333116289966) q[1];
ry(0.2902375942051092) q[3];
cx q[1],q[3];
ry(2.8302203493700056) q[1];
ry(2.6552021577828984) q[3];
cx q[1],q[3];
ry(-1.6126725038873464) q[0];
ry(-0.641077507510785) q[3];
cx q[0],q[3];
ry(-2.8726085199011946) q[0];
ry(2.0210067367128186) q[3];
cx q[0],q[3];
ry(2.4841335572000816) q[1];
ry(-1.7821036096674234) q[2];
cx q[1],q[2];
ry(-1.9339569894121826) q[1];
ry(1.771064973235436) q[2];
cx q[1],q[2];
ry(-1.5671279178631514) q[0];
ry(-2.3900122610256127) q[1];
cx q[0],q[1];
ry(-2.66036645426658) q[0];
ry(0.8634190688348697) q[1];
cx q[0],q[1];
ry(-1.9822716588300873) q[2];
ry(2.9618810053062403) q[3];
cx q[2],q[3];
ry(-1.2903837063393038) q[2];
ry(0.20813703780542614) q[3];
cx q[2],q[3];
ry(2.692027945601316) q[0];
ry(1.6253275273576806) q[2];
cx q[0],q[2];
ry(0.23521219739529647) q[0];
ry(-0.4391010380231748) q[2];
cx q[0],q[2];
ry(-2.5944178178240254) q[1];
ry(0.42266246722884854) q[3];
cx q[1],q[3];
ry(-3.0927697859566057) q[1];
ry(-0.11456697025605707) q[3];
cx q[1],q[3];
ry(-2.8716111722586892) q[0];
ry(-0.5741565827470891) q[3];
cx q[0],q[3];
ry(1.8991340571663475) q[0];
ry(-0.8117177059228267) q[3];
cx q[0],q[3];
ry(-0.9672179312442515) q[1];
ry(1.4898079972377327) q[2];
cx q[1],q[2];
ry(1.2908374021034659) q[1];
ry(1.1699686088886292) q[2];
cx q[1],q[2];
ry(2.159521504675732) q[0];
ry(-3.118381871289063) q[1];
cx q[0],q[1];
ry(-0.5472322262671059) q[0];
ry(-1.2755706383578982) q[1];
cx q[0],q[1];
ry(3.07313683752446) q[2];
ry(2.9499043981272792) q[3];
cx q[2],q[3];
ry(0.9737187440588952) q[2];
ry(0.15984650858837846) q[3];
cx q[2],q[3];
ry(0.7138778869888496) q[0];
ry(1.973070398868397) q[2];
cx q[0],q[2];
ry(0.11181480543123119) q[0];
ry(2.066474176592492) q[2];
cx q[0],q[2];
ry(-1.630316116919152) q[1];
ry(2.0158769791960482) q[3];
cx q[1],q[3];
ry(-1.8915065638832689) q[1];
ry(-2.2501279465741355) q[3];
cx q[1],q[3];
ry(-0.5318403539770866) q[0];
ry(0.584707061114532) q[3];
cx q[0],q[3];
ry(2.052447735690227) q[0];
ry(2.142995220454128) q[3];
cx q[0],q[3];
ry(0.972968783658227) q[1];
ry(-2.4186914770083923) q[2];
cx q[1],q[2];
ry(0.038838786322417704) q[1];
ry(-0.6415111886847535) q[2];
cx q[1],q[2];
ry(-2.403858236764897) q[0];
ry(-1.562363114234376) q[1];
cx q[0],q[1];
ry(2.8013083776564898) q[0];
ry(2.7372045668110427) q[1];
cx q[0],q[1];
ry(0.7180328744047149) q[2];
ry(-1.4996291539777085) q[3];
cx q[2],q[3];
ry(-0.8870227621879598) q[2];
ry(0.18274702582679822) q[3];
cx q[2],q[3];
ry(1.7096283028939618) q[0];
ry(1.5155889988355462) q[2];
cx q[0],q[2];
ry(2.06387939132874) q[0];
ry(0.22337044414079207) q[2];
cx q[0],q[2];
ry(2.535216758864668) q[1];
ry(-1.0671456875866054) q[3];
cx q[1],q[3];
ry(-1.930250653156778) q[1];
ry(2.384369434742547) q[3];
cx q[1],q[3];
ry(-2.3504092477252825) q[0];
ry(-1.8092016212855833) q[3];
cx q[0],q[3];
ry(-2.6816028092770154) q[0];
ry(-1.6647625905917183) q[3];
cx q[0],q[3];
ry(-0.6851108262876044) q[1];
ry(-0.31968305912030726) q[2];
cx q[1],q[2];
ry(-2.1303283326733675) q[1];
ry(0.9510527449896405) q[2];
cx q[1],q[2];
ry(1.488898379961191) q[0];
ry(-2.306129427436295) q[1];
ry(-0.6499252720167585) q[2];
ry(-0.4006893621170281) q[3];