OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(1.9740223794159757) q[0];
ry(-2.607914246932759) q[1];
cx q[0],q[1];
ry(0.9066416816503757) q[0];
ry(2.585456952298565) q[1];
cx q[0],q[1];
ry(-0.9412252939651458) q[2];
ry(-1.553184614556024) q[3];
cx q[2],q[3];
ry(-1.1289306198391253) q[2];
ry(3.091064528780465) q[3];
cx q[2],q[3];
ry(1.8162525966851781) q[4];
ry(-0.48060348441366685) q[5];
cx q[4],q[5];
ry(-2.4022882610383878) q[4];
ry(-2.015931924366673) q[5];
cx q[4],q[5];
ry(0.5738485653632397) q[6];
ry(-0.2399661708031182) q[7];
cx q[6],q[7];
ry(-1.1871531053785107) q[6];
ry(0.32765577644180865) q[7];
cx q[6],q[7];
ry(2.2354738818338644) q[8];
ry(-1.0922956203272403) q[9];
cx q[8],q[9];
ry(1.4029059769456698) q[8];
ry(0.6152661335025078) q[9];
cx q[8],q[9];
ry(1.7149262246719827) q[10];
ry(0.37163518034735693) q[11];
cx q[10],q[11];
ry(-1.4547187967111948) q[10];
ry(-1.9490249835530296) q[11];
cx q[10],q[11];
ry(-0.029438318918208292) q[0];
ry(0.44595338513803195) q[2];
cx q[0],q[2];
ry(1.6302651952770009) q[0];
ry(1.4660207388490814) q[2];
cx q[0],q[2];
ry(-2.750971126190334) q[2];
ry(0.11683544121387532) q[4];
cx q[2],q[4];
ry(-0.3788029073015826) q[2];
ry(0.8722688850165383) q[4];
cx q[2],q[4];
ry(1.2017659307957722) q[4];
ry(0.10324852931336274) q[6];
cx q[4],q[6];
ry(-2.808268248468582) q[4];
ry(-2.1706826805076957) q[6];
cx q[4],q[6];
ry(-0.2284538496181389) q[6];
ry(-2.5313650836825254) q[8];
cx q[6],q[8];
ry(-3.0523067359511193) q[6];
ry(-0.11914942705979215) q[8];
cx q[6],q[8];
ry(2.8924538229826617) q[8];
ry(-0.40160927428536297) q[10];
cx q[8],q[10];
ry(-1.9767080019794632) q[8];
ry(3.135352758098492) q[10];
cx q[8],q[10];
ry(2.4855512979002925) q[1];
ry(-1.5228486740153215) q[3];
cx q[1],q[3];
ry(1.6442153592734756) q[1];
ry(-1.5423409953129346) q[3];
cx q[1],q[3];
ry(2.378000269258816) q[3];
ry(-1.6573846697421377) q[5];
cx q[3],q[5];
ry(1.9176931302755167) q[3];
ry(-1.731464027235497) q[5];
cx q[3],q[5];
ry(2.8982186498556652) q[5];
ry(-2.2215049360717636) q[7];
cx q[5],q[7];
ry(-0.22620526553571035) q[5];
ry(-1.3143719611466984) q[7];
cx q[5],q[7];
ry(-0.5466363531947538) q[7];
ry(-0.4790517902026741) q[9];
cx q[7],q[9];
ry(-2.928634005555481) q[7];
ry(-3.0777249613016724) q[9];
cx q[7],q[9];
ry(-3.114659601176236) q[9];
ry(0.26569246333320606) q[11];
cx q[9],q[11];
ry(2.9301887039128824) q[9];
ry(0.1898750733716259) q[11];
cx q[9],q[11];
ry(-0.4879644480346367) q[0];
ry(-2.429870449097021) q[1];
cx q[0],q[1];
ry(0.6044053467064355) q[0];
ry(2.946779930686644) q[1];
cx q[0],q[1];
ry(1.427391060620388) q[2];
ry(2.133163672697699) q[3];
cx q[2],q[3];
ry(0.09618578559288068) q[2];
ry(-0.11569352299877343) q[3];
cx q[2],q[3];
ry(-0.6240284351299996) q[4];
ry(1.535404437014889) q[5];
cx q[4],q[5];
ry(1.8041918925697056) q[4];
ry(2.5784010853796584) q[5];
cx q[4],q[5];
ry(-0.7765007539828792) q[6];
ry(0.9587706592922758) q[7];
cx q[6],q[7];
ry(2.8585599090882896) q[6];
ry(2.715664795219326) q[7];
cx q[6],q[7];
ry(-1.2723586388962076) q[8];
ry(2.76266888795207) q[9];
cx q[8],q[9];
ry(1.0403329079842356) q[8];
ry(-3.1286078099355485) q[9];
cx q[8],q[9];
ry(1.6646623063531054) q[10];
ry(3.08290985899596) q[11];
cx q[10],q[11];
ry(1.4650633328813807) q[10];
ry(-2.6091327894905953) q[11];
cx q[10],q[11];
ry(0.5380678161613641) q[0];
ry(-0.5285080041706058) q[2];
cx q[0],q[2];
ry(0.7386084004380951) q[0];
ry(-3.018798586868276) q[2];
cx q[0],q[2];
ry(-1.7517712868186148) q[2];
ry(-0.5842273060049923) q[4];
cx q[2],q[4];
ry(-0.010116042592488661) q[2];
ry(0.09742934067188222) q[4];
cx q[2],q[4];
ry(-1.9067785903225651) q[4];
ry(-1.4306940681911717) q[6];
cx q[4],q[6];
ry(2.9655581143253484) q[4];
ry(-0.41953553397721377) q[6];
cx q[4],q[6];
ry(-0.864518105836904) q[6];
ry(0.5119077704989442) q[8];
cx q[6],q[8];
ry(3.1323210646665682) q[6];
ry(-3.040817595534539) q[8];
cx q[6],q[8];
ry(-0.19375372142164426) q[8];
ry(2.41431518210198) q[10];
cx q[8],q[10];
ry(0.4012641097994361) q[8];
ry(1.181924780002774) q[10];
cx q[8],q[10];
ry(-1.989903261412044) q[1];
ry(0.3558417873878154) q[3];
cx q[1],q[3];
ry(3.1218752234705613) q[1];
ry(-0.0012364235675896098) q[3];
cx q[1],q[3];
ry(-1.2540109039442915) q[3];
ry(-0.5043860628925647) q[5];
cx q[3],q[5];
ry(0.25830716941402354) q[3];
ry(-1.7960176019436196) q[5];
cx q[3],q[5];
ry(2.948660785279725) q[5];
ry(0.34093537138798613) q[7];
cx q[5],q[7];
ry(0.31179289898351925) q[5];
ry(-2.942774715708337) q[7];
cx q[5],q[7];
ry(0.6627050639513369) q[7];
ry(0.9665647087720464) q[9];
cx q[7],q[9];
ry(-0.6856610587963696) q[7];
ry(-2.2392423699910573) q[9];
cx q[7],q[9];
ry(1.497610486428047) q[9];
ry(2.3284869721065395) q[11];
cx q[9],q[11];
ry(-0.009726635491309922) q[9];
ry(-2.253827164732166) q[11];
cx q[9],q[11];
ry(-1.355168222746845) q[0];
ry(1.9800389557657418) q[1];
cx q[0],q[1];
ry(-0.12403663830547185) q[0];
ry(-2.885005626395852) q[1];
cx q[0],q[1];
ry(-0.8023972586660201) q[2];
ry(0.46397717300650854) q[3];
cx q[2],q[3];
ry(-0.6601096563193387) q[2];
ry(1.246466201831901) q[3];
cx q[2],q[3];
ry(0.22095555956928603) q[4];
ry(2.562208721455712) q[5];
cx q[4],q[5];
ry(2.8266963015521815) q[4];
ry(1.6522300413608022) q[5];
cx q[4],q[5];
ry(2.336649499465674) q[6];
ry(-0.3931404857231957) q[7];
cx q[6],q[7];
ry(3.0954421556658698) q[6];
ry(-3.1390631909396998) q[7];
cx q[6],q[7];
ry(-0.5693466214870244) q[8];
ry(1.579734926632442) q[9];
cx q[8],q[9];
ry(0.7931937797966436) q[8];
ry(-3.1062847765873993) q[9];
cx q[8],q[9];
ry(-0.8960259358844578) q[10];
ry(0.36027033171545675) q[11];
cx q[10],q[11];
ry(-2.5695520470640276) q[10];
ry(-2.4697102886400377) q[11];
cx q[10],q[11];
ry(-1.9586939149365183) q[0];
ry(0.8383493714349219) q[2];
cx q[0],q[2];
ry(-1.4106707429124814) q[0];
ry(1.0892578369096357) q[2];
cx q[0],q[2];
ry(0.25672219132521124) q[2];
ry(-1.9250145281726179) q[4];
cx q[2],q[4];
ry(0.07410551796320863) q[2];
ry(-3.1365030357330386) q[4];
cx q[2],q[4];
ry(1.7290723713053353) q[4];
ry(-0.3938044721710302) q[6];
cx q[4],q[6];
ry(-0.3015500808845601) q[4];
ry(1.819532378149895) q[6];
cx q[4],q[6];
ry(-1.3847424728333486) q[6];
ry(-0.7962775495985759) q[8];
cx q[6],q[8];
ry(0.00794618782086155) q[6];
ry(-3.136852226997536) q[8];
cx q[6],q[8];
ry(2.560843293093957) q[8];
ry(2.7306113956348708) q[10];
cx q[8],q[10];
ry(-0.34808531761551814) q[8];
ry(1.4425332107526678) q[10];
cx q[8],q[10];
ry(0.318221942544292) q[1];
ry(3.080099446353525) q[3];
cx q[1],q[3];
ry(-0.06338164179123607) q[1];
ry(1.7465606022405078) q[3];
cx q[1],q[3];
ry(-1.282077956012281) q[3];
ry(-1.8479830875727243) q[5];
cx q[3],q[5];
ry(-0.11921184835988452) q[3];
ry(3.125852909047126) q[5];
cx q[3],q[5];
ry(-0.6604509623867258) q[5];
ry(1.2693026822681464) q[7];
cx q[5],q[7];
ry(0.07981262180620338) q[5];
ry(0.05374044703482639) q[7];
cx q[5],q[7];
ry(1.381945361992785) q[7];
ry(-0.3025070450690821) q[9];
cx q[7],q[9];
ry(0.9184038390154727) q[7];
ry(-0.1120008629192375) q[9];
cx q[7],q[9];
ry(1.745724025290362) q[9];
ry(-0.41906633503063784) q[11];
cx q[9],q[11];
ry(0.0017041366291303817) q[9];
ry(-1.318656022761872) q[11];
cx q[9],q[11];
ry(2.2488284883695173) q[0];
ry(3.016339827784955) q[1];
cx q[0],q[1];
ry(-1.0695597582024021) q[0];
ry(1.7095585868457384) q[1];
cx q[0],q[1];
ry(1.032330471841166) q[2];
ry(1.234746610022869) q[3];
cx q[2],q[3];
ry(-3.0525358128541082) q[2];
ry(2.5458260418765315) q[3];
cx q[2],q[3];
ry(-1.9273470345279105) q[4];
ry(-1.3817187860189355) q[5];
cx q[4],q[5];
ry(1.4126804294909459) q[4];
ry(-1.5540642989885418) q[5];
cx q[4],q[5];
ry(-2.35442284966678) q[6];
ry(1.8114691286567712) q[7];
cx q[6],q[7];
ry(2.8235998245052607) q[6];
ry(-2.7936637540080564) q[7];
cx q[6],q[7];
ry(1.5702496884716526) q[8];
ry(-1.6018236567696968) q[9];
cx q[8],q[9];
ry(0.3691428133013046) q[8];
ry(-1.5765609926029187) q[9];
cx q[8],q[9];
ry(-0.7998144959921009) q[10];
ry(-1.7239568259070932) q[11];
cx q[10],q[11];
ry(2.2450166132996663) q[10];
ry(-2.7019402304422275) q[11];
cx q[10],q[11];
ry(-0.07390821618245681) q[0];
ry(2.428954398557739) q[2];
cx q[0],q[2];
ry(3.0048228667870047) q[0];
ry(-2.996852369045748) q[2];
cx q[0],q[2];
ry(-0.09006714548616923) q[2];
ry(1.5060278704754175) q[4];
cx q[2],q[4];
ry(-3.0069062216999045) q[2];
ry(2.9821454160670373) q[4];
cx q[2],q[4];
ry(-1.3934455759661928) q[4];
ry(1.1864083904401903) q[6];
cx q[4],q[6];
ry(0.06486197495753213) q[4];
ry(0.07255027956085157) q[6];
cx q[4],q[6];
ry(2.5918836065834205) q[6];
ry(0.5514812967963296) q[8];
cx q[6],q[8];
ry(3.1192257068101776) q[6];
ry(0.00030369606203972666) q[8];
cx q[6],q[8];
ry(-2.5755205714467664) q[8];
ry(1.7368706481720202) q[10];
cx q[8],q[10];
ry(1.556191865684506) q[8];
ry(1.1040609748062424) q[10];
cx q[8],q[10];
ry(1.5591836155973458) q[1];
ry(3.090628285296549) q[3];
cx q[1],q[3];
ry(-2.9050062786520963) q[1];
ry(1.483153615273313) q[3];
cx q[1],q[3];
ry(2.0520047783506183) q[3];
ry(1.7373478149092452) q[5];
cx q[3],q[5];
ry(-0.07544399190855694) q[3];
ry(3.0935846332830215) q[5];
cx q[3],q[5];
ry(1.8954465527397237) q[5];
ry(2.8118821853133786) q[7];
cx q[5],q[7];
ry(-0.022266887171668692) q[5];
ry(3.0900115541273685) q[7];
cx q[5],q[7];
ry(2.040400720442716) q[7];
ry(-2.985125260881576) q[9];
cx q[7],q[9];
ry(-3.0304724868624904) q[7];
ry(-3.0673564669738824) q[9];
cx q[7],q[9];
ry(0.1599683102468452) q[9];
ry(-1.1937210983911057) q[11];
cx q[9],q[11];
ry(1.5724201886474254) q[9];
ry(-0.4283002320814404) q[11];
cx q[9],q[11];
ry(1.534450415196038) q[0];
ry(-1.5388557706408426) q[1];
ry(1.698894162591567) q[2];
ry(-3.10462350104766) q[3];
ry(1.1392672838384172) q[4];
ry(0.609030958417898) q[5];
ry(-2.2236051787651965) q[6];
ry(1.5886081505912593) q[7];
ry(-1.5384891099126372) q[8];
ry(1.5611129650380327) q[9];
ry(1.5780684130021108) q[10];
ry(1.5556003181362463) q[11];