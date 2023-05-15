OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(1.1237324411002074) q[0];
rz(-1.4952764498634163) q[0];
ry(-2.304490705273641) q[1];
rz(-0.9799152714070098) q[1];
ry(2.020470799164142) q[2];
rz(-3.083095095086941) q[2];
ry(-2.108108909863933) q[3];
rz(-1.2269574886516847) q[3];
ry(1.5023977653693956) q[4];
rz(0.9313964368716676) q[4];
ry(1.540694050814067) q[5];
rz(2.912737448473844) q[5];
ry(-1.3186562234652717) q[6];
rz(1.4350395422702726) q[6];
ry(-1.1099089916511704) q[7];
rz(-0.3574055160436965) q[7];
ry(-2.8361336743599996) q[8];
rz(0.5731297849045571) q[8];
ry(-1.1192054648744287) q[9];
rz(-1.1819576702583765) q[9];
ry(-3.1303062813732754) q[10];
rz(-0.9191632519568653) q[10];
ry(0.0023208519310848885) q[11];
rz(-0.21902955188841777) q[11];
ry(1.403316783839943) q[12];
rz(0.055915993768626306) q[12];
ry(1.7857969801204823) q[13];
rz(-3.0959110937850327) q[13];
ry(3.135608225598731) q[14];
rz(-1.0181651456267584) q[14];
ry(3.119195114477611) q[15];
rz(-1.6881578070063383) q[15];
ry(-2.9292787060176684) q[16];
rz(-1.735611025125957) q[16];
ry(2.9422806883115524) q[17];
rz(1.864490930714659) q[17];
ry(-1.8125850301442599) q[18];
rz(1.5613465380830691) q[18];
ry(-1.6373853018018303) q[19];
rz(-2.6417733680068096) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.1101185294234055) q[0];
rz(-1.0418324846365528) q[0];
ry(-2.0296343297092037) q[1];
rz(-0.4495709378664375) q[1];
ry(-2.843682016255959) q[2];
rz(-0.380014740023799) q[2];
ry(2.1943249251407093) q[3];
rz(1.0080880691857192) q[3];
ry(-0.0028001103528181304) q[4];
rz(-0.42987557564392986) q[4];
ry(3.140851852607759) q[5];
rz(-0.16161018300105476) q[5];
ry(3.0956507631293495) q[6];
rz(-1.9282326521065505) q[6];
ry(-0.12858226769457123) q[7];
rz(-1.5178047770978265) q[7];
ry(2.9459571487511558) q[8];
rz(1.860145312732109) q[8];
ry(-0.05705329024020767) q[9];
rz(2.8504563890776855) q[9];
ry(0.08570642482272174) q[10];
rz(2.6929101248462173) q[10];
ry(-0.022676551828832103) q[11];
rz(-2.923167045642902) q[11];
ry(-2.327463107954124) q[12];
rz(-0.07971249893113619) q[12];
ry(2.124215765679532) q[13];
rz(-2.934498404051803) q[13];
ry(-0.5957309884101414) q[14];
rz(-2.9511253306875065) q[14];
ry(-0.19290347966457602) q[15];
rz(1.111523473701947) q[15];
ry(-1.6011999368470555) q[16];
rz(0.17493951717511558) q[16];
ry(1.7300442281792983) q[17];
rz(2.914747550660531) q[17];
ry(-1.2321199173638862) q[18];
rz(-1.6490928386017614) q[18];
ry(1.8785350441447832) q[19];
rz(2.7160029618111476) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(3.017871861432287) q[0];
rz(0.5834167776715061) q[0];
ry(1.6846610332331282) q[1];
rz(1.5792816180418896) q[1];
ry(1.7486042268403175) q[2];
rz(-0.05518248790034035) q[2];
ry(0.1935909933681649) q[3];
rz(-2.0402818305582118) q[3];
ry(0.10199032592815449) q[4];
rz(0.46932101847701374) q[4];
ry(-0.08824750157841521) q[5];
rz(2.3383462848338383) q[5];
ry(1.4365980491970305) q[6];
rz(2.2541866298527737) q[6];
ry(-0.7669508988590144) q[7];
rz(2.422827235318966) q[7];
ry(-1.5833446231215893) q[8];
rz(2.2260769569470202) q[8];
ry(-1.6931283696754709) q[9];
rz(-0.09387514784791706) q[9];
ry(-1.5520613174175044) q[10];
rz(-2.075238933621083) q[10];
ry(-1.601962938012533) q[11];
rz(1.3962540889122037) q[11];
ry(1.30056926679677) q[12];
rz(-1.3462827548848393) q[12];
ry(-1.9431665738193846) q[13];
rz(-1.0998822775234307) q[13];
ry(3.1184721829716104) q[14];
rz(0.3119298788471294) q[14];
ry(3.108165560862246) q[15];
rz(1.8735486685646556) q[15];
ry(-0.19908744201926715) q[16];
rz(-1.2267562267100076) q[16];
ry(-2.931416989105442) q[17];
rz(0.8352050554189084) q[17];
ry(-0.2944367222693074) q[18];
rz(-0.6303233559050072) q[18];
ry(-1.523701441224345) q[19];
rz(-0.7577051455317193) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.9886718868037114) q[0];
rz(-1.41532611024659) q[0];
ry(2.1535008100191018) q[1];
rz(-2.0080577767770373) q[1];
ry(1.3105710890911144) q[2];
rz(-0.606258788010589) q[2];
ry(1.8092467428606758) q[3];
rz(1.307848444243266) q[3];
ry(-3.1405976567363063) q[4];
rz(2.036657401578868) q[4];
ry(0.0032914788344164943) q[5];
rz(-1.8537620637709609) q[5];
ry(1.5422028284069267) q[6];
rz(1.9703158283436588) q[6];
ry(0.38833630804377434) q[7];
rz(0.4627163800886258) q[7];
ry(3.1343708253635985) q[8];
rz(0.947875781498031) q[8];
ry(-0.007641458687656311) q[9];
rz(2.9648698442090433) q[9];
ry(-0.00350034894930453) q[10];
rz(-2.84266424864715) q[10];
ry(3.123531523279688) q[11];
rz(1.2665488117957666) q[11];
ry(2.7304924550832146) q[12];
rz(1.2634267463147655) q[12];
ry(-2.444675647815208) q[13];
rz(2.6646586620080606) q[13];
ry(1.232814993939205) q[14];
rz(1.0302150318853724) q[14];
ry(2.565166030933307) q[15];
rz(2.864364585797671) q[15];
ry(-2.9565209401191077) q[16];
rz(-1.747772418098322) q[16];
ry(2.1730502090914268) q[17];
rz(0.7126208824058868) q[17];
ry(2.680021556485651) q[18];
rz(0.24380507085365985) q[18];
ry(-1.0914548034918008) q[19];
rz(-2.782819176254735) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.206812449778102) q[0];
rz(-0.9447036128487577) q[0];
ry(0.7465230353478294) q[1];
rz(3.088353692937983) q[1];
ry(1.0423242844963254) q[2];
rz(-1.403823495748633) q[2];
ry(1.058896126956974) q[3];
rz(1.2123400467194838) q[3];
ry(1.5735054657300624) q[4];
rz(-1.469672028190116) q[4];
ry(1.5666140845753995) q[5];
rz(1.8769960370971814) q[5];
ry(-3.1034194431588915) q[6];
rz(1.3352512454614767) q[6];
ry(-0.056283874805686196) q[7];
rz(2.425006220194992) q[7];
ry(-2.115822501402898) q[8];
rz(2.67552077170815) q[8];
ry(-0.030578219053940536) q[9];
rz(-1.2304524716564158) q[9];
ry(2.968844877394362) q[10];
rz(1.3260675170178633) q[10];
ry(3.082451111313117) q[11];
rz(-1.9494413001936028) q[11];
ry(0.10359328581704154) q[12];
rz(-1.5690877443471676) q[12];
ry(-3.1035952172030727) q[13];
rz(-0.48620190736150626) q[13];
ry(-3.1353639369201) q[14];
rz(2.032005994109917) q[14];
ry(0.006050639516133494) q[15];
rz(-0.059691931790532254) q[15];
ry(-0.635884839457499) q[16];
rz(-2.872577155605754) q[16];
ry(0.8220449443176827) q[17];
rz(0.4188631795929777) q[17];
ry(2.0120817609277384) q[18];
rz(-3.058194424432933) q[18];
ry(1.0890736819021853) q[19];
rz(-2.013319826884472) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.302316695394289) q[0];
rz(2.1401095453710273) q[0];
ry(-2.245169483422143) q[1];
rz(-2.6810248805127137) q[1];
ry(3.1240172704218883) q[2];
rz(-2.602538116492542) q[2];
ry(-0.011946976515372241) q[3];
rz(0.8116153433520134) q[3];
ry(0.0005137869262110684) q[4];
rz(-1.6688343417575062) q[4];
ry(3.1370845964740917) q[5];
rz(1.8794102019661805) q[5];
ry(3.0756784599133864) q[6];
rz(0.1076354553718728) q[6];
ry(1.5682202101176452) q[7];
rz(-1.0789313845074764) q[7];
ry(-0.0076545782002988005) q[8];
rz(1.5479074328695894) q[8];
ry(3.1219835959269164) q[9];
rz(-2.374185548548934) q[9];
ry(0.32991413200225145) q[10];
rz(0.1494370640326259) q[10];
ry(2.2033118836660033) q[11];
rz(1.2774749436767037) q[11];
ry(-2.660814711333091) q[12];
rz(-3.1162782303119423) q[12];
ry(2.5922797508189954) q[13];
rz(0.2266859310153322) q[13];
ry(-0.31536011453456764) q[14];
rz(1.7491844680899513) q[14];
ry(1.0812396111094997) q[15];
rz(0.8007863433044476) q[15];
ry(-0.6609476213920701) q[16];
rz(-0.2181736835934016) q[16];
ry(-0.8944002222402831) q[17];
rz(1.1446199379446984) q[17];
ry(-1.9590469019752916) q[18];
rz(-1.5315884996656266) q[18];
ry(-2.8572316821272863) q[19];
rz(0.04258922539005105) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-2.123960261698538) q[0];
rz(3.046862538204939) q[0];
ry(-0.7880583464693958) q[1];
rz(-0.2524452522609727) q[1];
ry(1.9512626880785282) q[2];
rz(0.19485551122980532) q[2];
ry(-1.211181997153954) q[3];
rz(2.3552397502812163) q[3];
ry(1.5654938928682818) q[4];
rz(2.356322039860273) q[4];
ry(-1.5685999442346694) q[5];
rz(-0.7427656364797871) q[5];
ry(-0.3409246395085201) q[6];
rz(1.7748955590228368) q[6];
ry(0.03373718413999901) q[7];
rz(-2.11208191661422) q[7];
ry(-1.71067197935463) q[8];
rz(-2.888042974295154) q[8];
ry(1.8327264782516821) q[9];
rz(3.035836857502828) q[9];
ry(1.5343037569675149) q[10];
rz(-1.6077652588007973) q[10];
ry(1.5443163097075772) q[11];
rz(-2.0924843368713963) q[11];
ry(3.101830501130899) q[12];
rz(1.9254862918755673) q[12];
ry(3.0930814877386297) q[13];
rz(-1.2618065451794875) q[13];
ry(-0.02859841531090712) q[14];
rz(1.1142948701600703) q[14];
ry(-0.023609000162743147) q[15];
rz(-0.7778832227001132) q[15];
ry(0.36959342433736087) q[16];
rz(3.1068275332798065) q[16];
ry(1.6578355287211268) q[17];
rz(-2.0749547426829738) q[17];
ry(-1.5338078144545362) q[18];
rz(0.4448730036748696) q[18];
ry(-0.1958858506970893) q[19];
rz(1.8873811607228559) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.297264463825458) q[0];
rz(2.109038169528289) q[0];
ry(-1.7909277378056552) q[1];
rz(3.015035983179003) q[1];
ry(-0.6563843384727485) q[2];
rz(-1.0998366613756607) q[2];
ry(1.0623277004368392) q[3];
rz(-2.426291111513987) q[3];
ry(-1.3252210968845848) q[4];
rz(2.1800972034563397) q[4];
ry(0.673562850867377) q[5];
rz(-3.014567385105764) q[5];
ry(1.1427960397028838) q[6];
rz(-0.5658721175143652) q[6];
ry(-1.6584245904910508) q[7];
rz(1.5716711927381537) q[7];
ry(-1.5593302566059055) q[8];
rz(0.4350402005453472) q[8];
ry(-1.575043638771831) q[9];
rz(0.1908553953897974) q[9];
ry(-2.067160386814529) q[10];
rz(-0.15633706635551972) q[10];
ry(0.7363394045216793) q[11];
rz(-0.14329833803041414) q[11];
ry(2.871955737624475) q[12];
rz(2.5252581387649284) q[12];
ry(0.11901680802084424) q[13];
rz(1.604557885117389) q[13];
ry(0.015823751699413722) q[14];
rz(-0.39391424466127684) q[14];
ry(-0.08870012879191816) q[15];
rz(-2.8268602927089836) q[15];
ry(-1.6445393045203254) q[16];
rz(-1.5995994341835285) q[16];
ry(0.4596795954461843) q[17];
rz(2.818697565595028) q[17];
ry(3.1372588926875014) q[18];
rz(-2.943437053361076) q[18];
ry(-0.021333353754735995) q[19];
rz(-0.8824923165405223) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-0.7554196470180865) q[0];
rz(-2.9812522438767743) q[0];
ry(1.2610559391468987) q[1];
rz(-2.530159634781308) q[1];
ry(0.0029224383251951645) q[2];
rz(-2.6279288822609184) q[2];
ry(3.1405490252314423) q[3];
rz(0.8012018662092566) q[3];
ry(0.011259278763033898) q[4];
rz(0.9513865147198262) q[4];
ry(-3.1402280361129464) q[5];
rz(0.13392209686175688) q[5];
ry(0.11642419046032604) q[6];
rz(1.6143645201342407) q[6];
ry(2.924771362352272) q[7];
rz(-1.539486873978771) q[7];
ry(0.22481461042849715) q[8];
rz(2.4733472525099125) q[8];
ry(-3.118910060180205) q[9];
rz(-0.7440742175863049) q[9];
ry(-0.495727391690669) q[10];
rz(0.24967284300212175) q[10];
ry(-1.7006788537578634) q[11];
rz(-2.44276719778168) q[11];
ry(-0.01515679071146436) q[12];
rz(0.6272685503836062) q[12];
ry(-3.132713861494584) q[13];
rz(2.6354524560366466) q[13];
ry(0.03350554497734609) q[14];
rz(-0.39023981483403875) q[14];
ry(0.0050332782247783925) q[15];
rz(1.827494977485563) q[15];
ry(-1.7671402023416833) q[16];
rz(0.6146605759030725) q[16];
ry(-0.16170478494922882) q[17];
rz(1.351305034010415) q[17];
ry(1.725442662650556) q[18];
rz(2.0417993078941303) q[18];
ry(-0.9206917922857798) q[19];
rz(1.676289057833501) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.7021818361911523) q[0];
rz(2.4110525576853457) q[0];
ry(1.0071489873542367) q[1];
rz(-1.2779124734865386) q[1];
ry(2.305672972507052) q[2];
rz(-3.0157869052970545) q[2];
ry(-0.2816101885716398) q[3];
rz(2.3008252205071758) q[3];
ry(-1.8096990476830648) q[4];
rz(0.3015234085140763) q[4];
ry(-2.4597010841594966) q[5];
rz(2.8305719505975393) q[5];
ry(1.5824296833175113) q[6];
rz(2.281025567467677) q[6];
ry(1.481863986518377) q[7];
rz(-1.5201249456776953) q[7];
ry(0.0013091985414126486) q[8];
rz(-0.005703972383683713) q[8];
ry(-0.013235628090441747) q[9];
rz(1.016086401188348) q[9];
ry(3.134251642672209) q[10];
rz(0.9369125326505489) q[10];
ry(3.0967874554082906) q[11];
rz(-0.30219948809669717) q[11];
ry(0.8398249084262162) q[12];
rz(-0.24184753020195782) q[12];
ry(1.3018754375965778) q[13];
rz(2.7065654069420253) q[13];
ry(-0.4222824144689703) q[14];
rz(0.5776446971055833) q[14];
ry(-1.8813063348151218) q[15];
rz(1.9069999360083107) q[15];
ry(2.86711843782046) q[16];
rz(0.5422901761283935) q[16];
ry(-0.38028759589384076) q[17];
rz(-0.6980260235619783) q[17];
ry(-1.4498890582482806) q[18];
rz(1.3123490910329183) q[18];
ry(1.7046722981249598) q[19];
rz(-2.297239949536726) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(0.5052953390530082) q[0];
rz(2.1541267409291254) q[0];
ry(-1.2757795085903663) q[1];
rz(-2.401623493347847) q[1];
ry(3.0665932328643204) q[2];
rz(0.34619983365624485) q[2];
ry(2.2734036230830026) q[3];
rz(-2.014074708428844) q[3];
ry(-1.5777046979158589) q[4];
rz(-0.7319944635970396) q[4];
ry(1.57685455879718) q[5];
rz(1.597056283967668) q[5];
ry(-0.022759930868336508) q[6];
rz(-2.741934502770974) q[6];
ry(-1.5235748734412633) q[7];
rz(1.6654564901202902) q[7];
ry(-1.762540125118225) q[8];
rz(-1.9428475817283024) q[8];
ry(1.5463067479848611) q[9];
rz(2.1872615462031293) q[9];
ry(-3.13923286700388) q[10];
rz(1.3892837221379322) q[10];
ry(0.20074231781785548) q[11];
rz(0.9080941591029887) q[11];
ry(-1.57316227943973) q[12];
rz(1.5384312850875126) q[12];
ry(1.5702767347248963) q[13];
rz(1.5854212015408082) q[13];
ry(-0.3806276511429179) q[14];
rz(2.8174561599585317) q[14];
ry(0.03973829713797127) q[15];
rz(2.3966610609714563) q[15];
ry(0.95510864938768) q[16];
rz(-2.9826718646080335) q[16];
ry(1.535020911092203) q[17];
rz(-2.741755439954335) q[17];
ry(2.0360847027927957) q[18];
rz(-0.4575075385989491) q[18];
ry(2.9436482246587925) q[19];
rz(2.9990560451074466) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(3.109435925810506) q[0];
rz(0.04790269919262453) q[0];
ry(-3.0362276446630165) q[1];
rz(-1.9873526090593565) q[1];
ry(-3.1014418417798466) q[2];
rz(0.44245169006318985) q[2];
ry(0.9661319910700179) q[3];
rz(2.416595152056774) q[3];
ry(3.0232240586550154) q[4];
rz(-2.252062193205682) q[4];
ry(-3.056003308681994) q[5];
rz(3.134521077871315) q[5];
ry(1.6127781132120995) q[6];
rz(1.425063697251958) q[6];
ry(2.2044322586974228) q[7];
rz(1.4832652967291073) q[7];
ry(0.09643037211577088) q[8];
rz(-1.106215304825903) q[8];
ry(3.0531544178532783) q[9];
rz(-2.3099127444240826) q[9];
ry(-3.1198915813533836) q[10];
rz(-1.0236540872289703) q[10];
ry(0.013268051080847167) q[11];
rz(-2.3721279946076583) q[11];
ry(1.646868998291378) q[12];
rz(2.76846157413323) q[12];
ry(-0.22201926955806695) q[13];
rz(1.4483360366309732) q[13];
ry(-0.007352465930949787) q[14];
rz(2.299594070563039) q[14];
ry(0.024762391637806032) q[15];
rz(-0.5594231976168055) q[15];
ry(2.894819123796367) q[16];
rz(-2.9631101406173044) q[16];
ry(3.0693860719486277) q[17];
rz(-2.762624562396604) q[17];
ry(2.2500206736069335) q[18];
rz(-1.15454798510638) q[18];
ry(0.07269236082660721) q[19];
rz(2.8848484616174703) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(3.100966656430971) q[0];
rz(1.8678187210188169) q[0];
ry(0.0969765579299322) q[1];
rz(0.776395028278885) q[1];
ry(0.12946196687898187) q[2];
rz(-2.759744056235441) q[2];
ry(-0.8034774784828785) q[3];
rz(-2.6398333182293383) q[3];
ry(-3.130712680906138) q[4];
rz(1.6286240871291726) q[4];
ry(-0.18176052027251455) q[5];
rz(1.5844949198941185) q[5];
ry(1.578720612958183) q[6];
rz(-3.1176019757432116) q[6];
ry(-1.6032977961520196) q[7];
rz(-3.141378530402903) q[7];
ry(-1.0085577986317) q[8];
rz(-1.114423492903253) q[8];
ry(3.0882314251684684) q[9];
rz(0.15499363804223354) q[9];
ry(-3.1086178456116835) q[10];
rz(-1.2716907747486514) q[10];
ry(-2.9345324188737107) q[11];
rz(-1.9196593551560832) q[11];
ry(-3.1222933055077866) q[12];
rz(1.1967551027565266) q[12];
ry(3.14153816625828) q[13];
rz(3.0222238613397368) q[13];
ry(0.3128223987087278) q[14];
rz(-0.17304124662376452) q[14];
ry(0.009033230851802188) q[15];
rz(-0.9815260298927856) q[15];
ry(2.3278616860775703) q[16];
rz(1.2108847272598364) q[16];
ry(0.3862647889190017) q[17];
rz(1.0408100687167556) q[17];
ry(1.7621208688814658) q[18];
rz(1.517406476808644) q[18];
ry(-0.11611292443943833) q[19];
rz(1.8413072342613968) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(3.1195567535026076) q[0];
rz(0.12071362719720823) q[0];
ry(-2.958538643478462) q[1];
rz(1.8017507505049526) q[1];
ry(3.094981481968684) q[2];
rz(-2.196325901031852) q[2];
ry(2.593552313471665) q[3];
rz(-2.4150658071377276) q[3];
ry(1.5701435533779344) q[4];
rz(2.712212015505207) q[4];
ry(1.5864129862232208) q[5];
rz(0.4978401824964591) q[5];
ry(1.5734020017860237) q[6];
rz(2.7010327647915116) q[6];
ry(1.5725282735657355) q[7];
rz(3.123333075952202) q[7];
ry(-0.02895203552056902) q[8];
rz(-2.346256929332469) q[8];
ry(0.014838697856911054) q[9];
rz(-1.8343614121950285) q[9];
ry(-3.1075573384726005) q[10];
rz(-2.89389239086786) q[10];
ry(0.015925272896978093) q[11];
rz(-0.5933477250048489) q[11];
ry(-1.584787775221431) q[12];
rz(-1.3546784006043726) q[12];
ry(-1.5657783726200833) q[13];
rz(0.3019960712440454) q[13];
ry(0.007701458034375311) q[14];
rz(-1.1701496147946617) q[14];
ry(-3.129254495569017) q[15];
rz(0.9521861457427174) q[15];
ry(3.118484870480064) q[16];
rz(-0.624711024019008) q[16];
ry(-0.10144203868140522) q[17];
rz(0.44644771935068944) q[17];
ry(-1.5538976771158952) q[18];
rz(1.6302861312396724) q[18];
ry(3.113315206713664) q[19];
rz(1.585663194262545) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-0.059399916502218986) q[0];
rz(2.244016487447714) q[0];
ry(-2.9969347208983037) q[1];
rz(-1.3408600942497995) q[1];
ry(-3.1266791107816494) q[2];
rz(-2.4975463271197467) q[2];
ry(-2.4150884250968394) q[3];
rz(-1.9379986841619923) q[3];
ry(-1.007910423141979) q[4];
rz(0.7824361428752509) q[4];
ry(-1.7569685204522585) q[5];
rz(0.8359922153055728) q[5];
ry(-1.764389362672353) q[6];
rz(1.1553313543138488) q[6];
ry(1.7559834718828675) q[7];
rz(2.7162812581741558) q[7];
ry(0.4558284141587565) q[8];
rz(2.973095910314388) q[8];
ry(2.0638970228614975) q[9];
rz(2.2747934700373547) q[9];
ry(-2.4563781568319887) q[10];
rz(2.9216706409462083) q[10];
ry(-0.3199883943582833) q[11];
rz(3.0716134712962417) q[11];
ry(2.732409808533223) q[12];
rz(-0.803793555868012) q[12];
ry(-1.3459258504059939) q[13];
rz(-0.5768413421525097) q[13];
ry(0.07683064949846127) q[14];
rz(1.8408373343455287) q[14];
ry(-1.8823840510915741) q[15];
rz(-0.3550887156706395) q[15];
ry(-1.4511394853702968) q[16];
rz(-1.2577057645060372) q[16];
ry(1.5169607783491568) q[17];
rz(2.0847510434683274) q[17];
ry(1.2442298232661475) q[18];
rz(3.0480712961817185) q[18];
ry(-0.12671528471611093) q[19];
rz(-0.8874829697638614) q[19];