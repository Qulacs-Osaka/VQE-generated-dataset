OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(1.4313979158086356) q[0];
rz(-0.0004039363001967415) q[0];
ry(1.8661728687732926) q[1];
rz(3.1349878541521106) q[1];
ry(2.0817285100195182e-05) q[2];
rz(-1.359621686733383) q[2];
ry(-0.8521333183821188) q[3];
rz(1.246169353779526) q[3];
ry(3.1394080941343376) q[4];
rz(0.7717466834276169) q[4];
ry(1.5701657715768444) q[5];
rz(-0.0002172397824864447) q[5];
ry(1.5703407200917643) q[6];
rz(-0.9956170826797591) q[6];
ry(-1.5717615764222919) q[7];
rz(0.08195829736506713) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-1.8897235617417811) q[0];
rz(-1.5672614235783235) q[0];
ry(1.0378625478979178) q[1];
rz(-0.1495783115077165) q[1];
ry(-3.1415579032524916) q[2];
rz(-0.2520285732238952) q[2];
ry(-0.00046943292778767187) q[3];
rz(1.8821224254301294) q[3];
ry(1.4049846558574792) q[4];
rz(-0.30037605653528965) q[4];
ry(-2.739777870548804) q[5];
rz(2.903695716414101) q[5];
ry(-0.16607048918933565) q[6];
rz(-0.5060782337551641) q[6];
ry(-0.059567121437100246) q[7];
rz(1.2098620520687708) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-1.564860753029814) q[0];
rz(-2.8517546132987484) q[0];
ry(-0.18750848670382766) q[1];
rz(-2.9884542085720254) q[1];
ry(-1.5707932777677147) q[2];
rz(1.5702578849003639) q[2];
ry(2.267618601215392) q[3];
rz(-1.5676591574643404) q[3];
ry(-3.1386384105334013) q[4];
rz(-0.5221603594147894) q[4];
ry(3.062281751092283e-05) q[5];
rz(0.23692700029446254) q[5];
ry(-3.138240528287196) q[6];
rz(-1.4348260066535723) q[6];
ry(-0.005856039694431617) q[7];
rz(-1.158476569834671) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-1.5708028356220751) q[0];
rz(3.1411339145255277) q[0];
ry(-1.5799016438242939) q[1];
rz(-1.2049521105254828) q[1];
ry(-0.5168242278131912) q[2];
rz(-0.44808589887152844) q[2];
ry(1.5708113014501703) q[3];
rz(1.5708460829085735) q[3];
ry(3.095938192607487) q[4];
rz(2.1089977320108204) q[4];
ry(-0.33764018648467786) q[5];
rz(0.2527491813007464) q[5];
ry(1.4285407773107395) q[6];
rz(2.4657788971131547) q[6];
ry(1.4022075527798332) q[7];
rz(0.3684787149449819) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-0.2853067366391884) q[0];
rz(-3.1410965689420567) q[0];
ry(-1.57071484594917) q[1];
rz(-2.8434968006797976) q[1];
ry(0.00039814551540979295) q[2];
rz(-2.693703594396688) q[2];
ry(-1.570777071198557) q[3];
rz(-0.587723872789731) q[3];
ry(-0.008187742807443112) q[4];
rz(-2.333792592428521) q[4];
ry(-3.141553826588918) q[5];
rz(-0.4951048750526974) q[5];
ry(3.13993804001365) q[6];
rz(-2.865518196935203) q[6];
ry(-3.1162301588766357) q[7];
rz(2.761537464907325) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-1.5708029929235132) q[0];
rz(0.8598467388539005) q[0];
ry(3.1415780425448543) q[1];
rz(-1.5107877355144652) q[1];
ry(1.5708016110793386) q[2];
rz(1.5695691455224645) q[2];
ry(1.5705456663153885) q[3];
rz(3.080364385931048) q[3];
ry(1.6152916813264424) q[4];
rz(3.1320497852532094) q[4];
ry(-5.007583535395839e-06) q[5];
rz(-0.5441746555979579) q[5];
ry(3.052003906700579) q[6];
rz(1.837478477748364) q[6];
ry(-0.14966659983891528) q[7];
rz(0.7445334518770359) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-1.5089238977417097e-05) q[0];
rz(0.7107515221012708) q[0];
ry(5.867809404991591e-05) q[1];
rz(3.107084869932717) q[1];
ry(-1.8519951164140955) q[2];
rz(-0.0016304348483944804) q[2];
ry(0.04047359494988662) q[3];
rz(0.46037976185987795) q[3];
ry(1.5707541789011081) q[4];
rz(1.343276575221899) q[4];
ry(1.5708645408685906) q[5];
rz(-0.0003657455896770825) q[5];
ry(-3.1415908589727812) q[6];
rz(2.2565536929364427) q[6];
ry(-1.5934448427352619) q[7];
rz(2.961497976953329) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-1.5708384982715922) q[0];
rz(1.570789664326127) q[0];
ry(-3.1415784025065463) q[1];
rz(1.2982079954601509) q[1];
ry(-1.5722031732443495) q[2];
rz(0.03082301276692867) q[2];
ry(1.5705648288452734) q[3];
rz(0.7130963031925079) q[3];
ry(1.5707814195780263) q[4];
rz(-1.5707991001479) q[4];
ry(-1.5709144729054865) q[5];
rz(2.777924677013694) q[5];
ry(-3.1415835666621854) q[6];
rz(0.18904760078839794) q[6];
ry(3.1369223451620742) q[7];
rz(-1.4820296433673041) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-2.39781814590306) q[0];
rz(-1.2559551466108172) q[0];
ry(1.5707160506258981) q[1];
rz(-1.6566823966743194) q[1];
ry(-3.1415438768064323) q[2];
rz(0.7772197685263836) q[2];
ry(-3.140541542432801) q[3];
rz(-0.8577128435313727) q[3];
ry(0.41005065294336834) q[4];
rz(-1.5707561624273625) q[4];
ry(9.306194537643229e-06) q[5];
rz(-2.779891480095503) q[5];
ry(-3.14110842055946) q[6];
rz(-2.1246847890187457) q[6];
ry(-2.7577850517795595) q[7];
rz(1.1220902818179461) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-6.344068315300433e-05) q[0];
rz(2.826749425693434) q[0];
ry(-3.125756135926208e-05) q[1];
rz(0.08629156477783263) q[1];
ry(-1.5714895264598134) q[2];
rz(-2.172873770661554) q[2];
ry(1.5708350425857378) q[3];
rz(-1.2590811550218954) q[3];
ry(1.57077364965658) q[4];
rz(0.2788322979393749) q[4];
ry(-1.5702175439420831) q[5];
rz(-0.000697908399985859) q[5];
ry(-1.2883513569761362e-05) q[6];
rz(-1.078170185264217) q[6];
ry(-3.1401654690960443) q[7];
rz(-2.923569093217958) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-1.5707933066003346) q[0];
rz(-0.14499586831399555) q[0];
ry(1.57100952561691) q[1];
rz(-1.062754238488034) q[1];
ry(3.1415395817199196) q[2];
rz(2.3935930961565077) q[2];
ry(-0.00010521505086469318) q[3];
rz(2.166648345370849) q[3];
ry(-3.1415694587437097) q[4];
rz(-1.6669088655474837) q[4];
ry(1.5709850953907802) q[5];
rz(-2.234401488859186) q[5];
ry(3.1415026338697585) q[6];
rz(-2.269727165658627) q[6];
ry(-0.00016473140035166978) q[7];
rz(1.8124086955807508) q[7];