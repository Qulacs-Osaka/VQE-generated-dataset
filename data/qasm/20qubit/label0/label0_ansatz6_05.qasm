OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(2.2255306911975588) q[0];
ry(-0.36407299412944205) q[1];
cx q[0],q[1];
ry(-0.8638462107380329) q[0];
ry(-2.9084835900443444) q[1];
cx q[0],q[1];
ry(0.12321556908556135) q[1];
ry(-0.22655208667723753) q[2];
cx q[1],q[2];
ry(2.551979921934461) q[1];
ry(-3.06558141571149) q[2];
cx q[1],q[2];
ry(-1.5423490967012166) q[2];
ry(-2.2447885829153935) q[3];
cx q[2],q[3];
ry(2.279586869099955) q[2];
ry(0.876117937508914) q[3];
cx q[2],q[3];
ry(-0.051345361666674284) q[3];
ry(1.812079705317747) q[4];
cx q[3],q[4];
ry(-3.0391604421579044) q[3];
ry(-3.1102576208444055) q[4];
cx q[3],q[4];
ry(2.699770547384868) q[4];
ry(3.010232539855111) q[5];
cx q[4],q[5];
ry(1.0450366037610404) q[4];
ry(-1.0751713095304574) q[5];
cx q[4],q[5];
ry(0.825699913189411) q[5];
ry(2.225836573824384) q[6];
cx q[5],q[6];
ry(-3.1397272959777536) q[5];
ry(0.0056168379609451815) q[6];
cx q[5],q[6];
ry(1.7642013160853816) q[6];
ry(-1.0212891468077416) q[7];
cx q[6],q[7];
ry(2.380094958092269) q[6];
ry(-2.063131251519598) q[7];
cx q[6],q[7];
ry(-1.2387250509828098) q[7];
ry(-1.07605621787063) q[8];
cx q[7],q[8];
ry(-1.5515053261539529) q[7];
ry(-0.7328987061113557) q[8];
cx q[7],q[8];
ry(-2.8505808883173356) q[8];
ry(-1.25160737801213) q[9];
cx q[8],q[9];
ry(-0.737743600301014) q[8];
ry(-1.7084445348760897) q[9];
cx q[8],q[9];
ry(1.7882205020766968) q[9];
ry(1.5641760517962164) q[10];
cx q[9],q[10];
ry(0.2298457599233874) q[9];
ry(-3.139361473875935) q[10];
cx q[9],q[10];
ry(-1.5379547705546024) q[10];
ry(-1.4361606594771865) q[11];
cx q[10],q[11];
ry(0.025079934384418082) q[10];
ry(1.5511049774372565) q[11];
cx q[10],q[11];
ry(-1.081874576781333) q[11];
ry(-2.693237849745214) q[12];
cx q[11],q[12];
ry(0.05228429863520124) q[11];
ry(-0.05737619359228948) q[12];
cx q[11],q[12];
ry(-1.959517367182336) q[12];
ry(-2.979815310156103) q[13];
cx q[12],q[13];
ry(0.18583398897454906) q[12];
ry(2.9567923381223826) q[13];
cx q[12],q[13];
ry(0.22321617161158686) q[13];
ry(0.04754170004007641) q[14];
cx q[13],q[14];
ry(-0.8577162999574517) q[13];
ry(-1.3296037701300571) q[14];
cx q[13],q[14];
ry(2.29116421208422) q[14];
ry(-1.5733968079995604) q[15];
cx q[14],q[15];
ry(-1.4401075531913374) q[14];
ry(0.011937269440590743) q[15];
cx q[14],q[15];
ry(-0.5095297727558795) q[15];
ry(-1.6587041957954682) q[16];
cx q[15],q[16];
ry(1.0948624713450874) q[15];
ry(-0.9619548437905028) q[16];
cx q[15],q[16];
ry(0.6015014356351678) q[16];
ry(-2.1975571386556356) q[17];
cx q[16],q[17];
ry(-3.1412954750454847) q[16];
ry(0.00265758381496954) q[17];
cx q[16],q[17];
ry(0.5845041166641289) q[17];
ry(2.7112619733107253) q[18];
cx q[17],q[18];
ry(0.5975436039009098) q[17];
ry(-1.7130917078048187) q[18];
cx q[17],q[18];
ry(0.17022082865422591) q[18];
ry(-2.179076842073766) q[19];
cx q[18],q[19];
ry(-2.9090998390317875) q[18];
ry(1.1676453622192986) q[19];
cx q[18],q[19];
ry(-2.1318206299270934) q[0];
ry(-2.579901602924821) q[1];
cx q[0],q[1];
ry(2.2986576533906606) q[0];
ry(2.364132297109341) q[1];
cx q[0],q[1];
ry(-2.0109915679051302) q[1];
ry(0.3721865591415615) q[2];
cx q[1],q[2];
ry(-0.12592888162443175) q[1];
ry(2.538522993014217) q[2];
cx q[1],q[2];
ry(-0.7902215514664235) q[2];
ry(-0.9761222514033928) q[3];
cx q[2],q[3];
ry(1.090306601616595) q[2];
ry(-0.031127462898885128) q[3];
cx q[2],q[3];
ry(2.702819992614376) q[3];
ry(0.9081320501205283) q[4];
cx q[3],q[4];
ry(-0.36350770732531407) q[3];
ry(3.1412380206824313) q[4];
cx q[3],q[4];
ry(2.115562406315589) q[4];
ry(-2.3858435326572494) q[5];
cx q[4],q[5];
ry(-0.1841538570886942) q[4];
ry(-3.0723966092382726) q[5];
cx q[4],q[5];
ry(-2.0503772518818706) q[5];
ry(-0.502140965485802) q[6];
cx q[5],q[6];
ry(-0.018073266698448616) q[5];
ry(3.1311440290423787) q[6];
cx q[5],q[6];
ry(1.721538656900197) q[6];
ry(0.6582520454082744) q[7];
cx q[6],q[7];
ry(-2.368452686919529) q[6];
ry(0.8948518266192552) q[7];
cx q[6],q[7];
ry(0.5045364846347953) q[7];
ry(-2.0167520911007655) q[8];
cx q[7],q[8];
ry(1.7002718334736864) q[7];
ry(-2.329322372073632) q[8];
cx q[7],q[8];
ry(-2.9441731386810774) q[8];
ry(1.0077222148487035) q[9];
cx q[8],q[9];
ry(1.0078014370549255) q[8];
ry(-3.0196735491942652) q[9];
cx q[8],q[9];
ry(-0.11919376054140347) q[9];
ry(-1.5856992807061427) q[10];
cx q[9],q[10];
ry(-0.3873920084078277) q[9];
ry(3.1353166992753416) q[10];
cx q[9],q[10];
ry(-2.4040488176974586) q[10];
ry(-1.6463632241347552) q[11];
cx q[10],q[11];
ry(-1.1968383995663787) q[10];
ry(2.2072230963776573) q[11];
cx q[10],q[11];
ry(1.0461827457930362) q[11];
ry(0.5469190501904029) q[12];
cx q[11],q[12];
ry(-1.0592943022104118) q[11];
ry(-2.9934208305031738) q[12];
cx q[11],q[12];
ry(0.08715647451326074) q[12];
ry(-1.2150293703478656) q[13];
cx q[12],q[13];
ry(-1.8436288481552054) q[12];
ry(-1.5593356984537845) q[13];
cx q[12],q[13];
ry(-2.2020912784418996) q[13];
ry(-1.7038287721639058) q[14];
cx q[13],q[14];
ry(2.8639843761102592) q[13];
ry(3.1178940434182048) q[14];
cx q[13],q[14];
ry(-2.0353556275866582) q[14];
ry(-1.517284245702019) q[15];
cx q[14],q[15];
ry(0.02067734587262038) q[14];
ry(3.1386087516067884) q[15];
cx q[14],q[15];
ry(-1.495501009609875) q[15];
ry(-2.204446339092886) q[16];
cx q[15],q[16];
ry(2.185912761207166) q[15];
ry(2.4382246519585835) q[16];
cx q[15],q[16];
ry(1.3322023275961286) q[16];
ry(-2.4322148528850818) q[17];
cx q[16],q[17];
ry(3.0099439228511398) q[16];
ry(-0.258039073857868) q[17];
cx q[16],q[17];
ry(2.3295638870028057) q[17];
ry(-2.1318398931343596) q[18];
cx q[17],q[18];
ry(0.0003973308083002891) q[17];
ry(0.0002550629773923416) q[18];
cx q[17],q[18];
ry(0.014552657383872977) q[18];
ry(-2.7659987968031072) q[19];
cx q[18],q[19];
ry(-0.341209207018184) q[18];
ry(2.0584065296919105) q[19];
cx q[18],q[19];
ry(-2.2978402992479956) q[0];
ry(-1.360580431364391) q[1];
cx q[0],q[1];
ry(2.1292756396934616) q[0];
ry(-0.7569358694980544) q[1];
cx q[0],q[1];
ry(1.3584630618142162) q[1];
ry(-2.418017262690353) q[2];
cx q[1],q[2];
ry(3.1001358219463806) q[1];
ry(-2.919292523872953) q[2];
cx q[1],q[2];
ry(0.9328116830763005) q[2];
ry(-1.088495294539141) q[3];
cx q[2],q[3];
ry(-2.5658398060254695) q[2];
ry(2.6061734335107536) q[3];
cx q[2],q[3];
ry(-2.8502607814796175) q[3];
ry(-2.663771935825789) q[4];
cx q[3],q[4];
ry(0.18648064765986572) q[3];
ry(-2.818518450853666) q[4];
cx q[3],q[4];
ry(1.502311792415158) q[4];
ry(2.212178949289404) q[5];
cx q[4],q[5];
ry(-2.132270341525272) q[4];
ry(0.007685837954168662) q[5];
cx q[4],q[5];
ry(1.6749292518859944) q[5];
ry(-1.4415916639413737) q[6];
cx q[5],q[6];
ry(0.005330822098926303) q[5];
ry(-2.948510616948399) q[6];
cx q[5],q[6];
ry(1.9450492263302293) q[6];
ry(-1.095996081040617) q[7];
cx q[6],q[7];
ry(-2.584965170977706) q[6];
ry(0.9418923564295762) q[7];
cx q[6],q[7];
ry(1.4069920486080232) q[7];
ry(-2.1048003640215276) q[8];
cx q[7],q[8];
ry(0.25422548969091885) q[7];
ry(2.7917909216557035) q[8];
cx q[7],q[8];
ry(-1.956778094788957) q[8];
ry(-2.9141314100891913) q[9];
cx q[8],q[9];
ry(1.738956431515101) q[8];
ry(0.45882351096773366) q[9];
cx q[8],q[9];
ry(-1.5941106113191506) q[9];
ry(1.5794512570111126) q[10];
cx q[9],q[10];
ry(1.1194425152584042) q[9];
ry(-3.1278174892692157) q[10];
cx q[9],q[10];
ry(0.5616300583961467) q[10];
ry(-0.7243250636085329) q[11];
cx q[10],q[11];
ry(-3.1386085075883137) q[10];
ry(-0.011424050210042584) q[11];
cx q[10],q[11];
ry(-1.6552109470583933) q[11];
ry(2.7383971675711747) q[12];
cx q[11],q[12];
ry(0.06360050002927747) q[11];
ry(3.0964855194043484) q[12];
cx q[11],q[12];
ry(-3.1031959004568224) q[12];
ry(2.218805960887354) q[13];
cx q[12],q[13];
ry(-3.09784957508086) q[12];
ry(1.5728841665406244) q[13];
cx q[12],q[13];
ry(2.5618454981973326) q[13];
ry(0.9349006650939347) q[14];
cx q[13],q[14];
ry(-0.25586925007751465) q[13];
ry(-3.128647578976911) q[14];
cx q[13],q[14];
ry(0.6706872924417944) q[14];
ry(0.9329066657063745) q[15];
cx q[14],q[15];
ry(-1.5542387517955072) q[14];
ry(-3.089939193352308) q[15];
cx q[14],q[15];
ry(-0.9218549809921657) q[15];
ry(-1.4049861722262582) q[16];
cx q[15],q[16];
ry(-2.2075001619586763) q[15];
ry(-3.1321636398498347) q[16];
cx q[15],q[16];
ry(-2.7440742836702547) q[16];
ry(-2.25962892804815) q[17];
cx q[16],q[17];
ry(-0.3398114471950233) q[16];
ry(2.8853742434134944) q[17];
cx q[16],q[17];
ry(2.1346081752439785) q[17];
ry(-0.019258603221044446) q[18];
cx q[17],q[18];
ry(2.2852293060232896) q[17];
ry(0.005297522906048017) q[18];
cx q[17],q[18];
ry(2.64304153419614) q[18];
ry(-0.05859507122830021) q[19];
cx q[18],q[19];
ry(0.8160283295349418) q[18];
ry(1.7383534651735266) q[19];
cx q[18],q[19];
ry(-0.10064321310678173) q[0];
ry(2.1898251039205068) q[1];
cx q[0],q[1];
ry(-2.9739394698727444) q[0];
ry(-0.6656971645488988) q[1];
cx q[0],q[1];
ry(1.2131007680977373) q[1];
ry(-0.16094015608612885) q[2];
cx q[1],q[2];
ry(0.35747481100964423) q[1];
ry(2.033742643283885) q[2];
cx q[1],q[2];
ry(-0.2682023439054152) q[2];
ry(0.641883336734784) q[3];
cx q[2],q[3];
ry(0.024774094972819682) q[2];
ry(-2.89980088140628) q[3];
cx q[2],q[3];
ry(-0.13888943417279087) q[3];
ry(0.9720546117869553) q[4];
cx q[3],q[4];
ry(0.005495665494900401) q[3];
ry(-0.10771838606386196) q[4];
cx q[3],q[4];
ry(-1.0426134749349363) q[4];
ry(1.5806588757778077) q[5];
cx q[4],q[5];
ry(1.1091237465399386) q[4];
ry(0.00028174982043062814) q[5];
cx q[4],q[5];
ry(-1.4746623911405805) q[5];
ry(-2.6246264503834165) q[6];
cx q[5],q[6];
ry(3.0776711891063706) q[5];
ry(1.8516835596065446) q[6];
cx q[5],q[6];
ry(-1.4955699360259496) q[6];
ry(0.18059212015570658) q[7];
cx q[6],q[7];
ry(0.42818890114457986) q[6];
ry(0.0030658953756885765) q[7];
cx q[6],q[7];
ry(-2.525977491101235) q[7];
ry(-1.7351469050065127) q[8];
cx q[7],q[8];
ry(3.120711509486326) q[7];
ry(-0.0015738135974165957) q[8];
cx q[7],q[8];
ry(-2.6506534888956006) q[8];
ry(-2.3537320917795346) q[9];
cx q[8],q[9];
ry(3.0637338473501226) q[8];
ry(-1.0727785223248538) q[9];
cx q[8],q[9];
ry(0.7754850846317594) q[9];
ry(1.2605683288001384) q[10];
cx q[9],q[10];
ry(1.3662172151563086) q[9];
ry(-0.03918900922278912) q[10];
cx q[9],q[10];
ry(-1.287003363960583) q[10];
ry(2.397744470218325) q[11];
cx q[10],q[11];
ry(-3.054997226260348) q[10];
ry(-0.6995977355765589) q[11];
cx q[10],q[11];
ry(2.40911536163391) q[11];
ry(-2.4733448105706057) q[12];
cx q[11],q[12];
ry(2.3884100248666567) q[11];
ry(3.137585477531623) q[12];
cx q[11],q[12];
ry(-0.8018080635392675) q[12];
ry(1.028229649882478) q[13];
cx q[12],q[13];
ry(-1.4637669300140461) q[12];
ry(0.1653497768188661) q[13];
cx q[12],q[13];
ry(2.3353921378985456) q[13];
ry(-2.3559434657386356) q[14];
cx q[13],q[14];
ry(0.008459081762743239) q[13];
ry(-0.1768984546883203) q[14];
cx q[13],q[14];
ry(2.249840314816354) q[14];
ry(2.907612277204513) q[15];
cx q[14],q[15];
ry(2.3752305647270866) q[14];
ry(-3.108037376195391) q[15];
cx q[14],q[15];
ry(1.6546952816944487) q[15];
ry(2.4015843986907512) q[16];
cx q[15],q[16];
ry(-0.7677733068010596) q[15];
ry(1.8393676546347797) q[16];
cx q[15],q[16];
ry(0.7586111239559763) q[16];
ry(2.5066037517783877) q[17];
cx q[16],q[17];
ry(2.999147531271675) q[16];
ry(-0.033537196824730664) q[17];
cx q[16],q[17];
ry(2.896415924363048) q[17];
ry(-2.039018022840337) q[18];
cx q[17],q[18];
ry(-0.7575684508315499) q[17];
ry(-2.4036358291362077) q[18];
cx q[17],q[18];
ry(-1.3948900436413283) q[18];
ry(2.2196236217239296) q[19];
cx q[18],q[19];
ry(-2.129860148266883) q[18];
ry(1.1776150841435233) q[19];
cx q[18],q[19];
ry(-0.2892200562387153) q[0];
ry(2.5915328403497617) q[1];
cx q[0],q[1];
ry(-0.19460698691761902) q[0];
ry(2.926865337526168) q[1];
cx q[0],q[1];
ry(-1.384686433934764) q[1];
ry(3.090520792871798) q[2];
cx q[1],q[2];
ry(0.2529681341158465) q[1];
ry(-2.8667980178490264) q[2];
cx q[1],q[2];
ry(-1.3374280162917644) q[2];
ry(0.5932370338926409) q[3];
cx q[2],q[3];
ry(2.8637204101194413) q[2];
ry(0.8608429774458788) q[3];
cx q[2],q[3];
ry(2.721317781240867) q[3];
ry(0.396679972466885) q[4];
cx q[3],q[4];
ry(0.02861559962899545) q[3];
ry(-0.10313311917502073) q[4];
cx q[3],q[4];
ry(-0.4894326522240524) q[4];
ry(-0.9710142185573476) q[5];
cx q[4],q[5];
ry(2.7434329055604803) q[4];
ry(2.659313813158093e-05) q[5];
cx q[4],q[5];
ry(-1.6927526782280262) q[5];
ry(0.6981744499504047) q[6];
cx q[5],q[6];
ry(-3.0090524137488055) q[5];
ry(0.373500419603782) q[6];
cx q[5],q[6];
ry(0.28592147176791016) q[6];
ry(0.7338975019668859) q[7];
cx q[6],q[7];
ry(1.2834360596035186) q[6];
ry(-2.7010996301922674) q[7];
cx q[6],q[7];
ry(0.5697244510188002) q[7];
ry(-0.4211444053716047) q[8];
cx q[7],q[8];
ry(3.1242157679562967) q[7];
ry(0.8092254572835696) q[8];
cx q[7],q[8];
ry(-0.2096256019319297) q[8];
ry(2.455832918449302) q[9];
cx q[8],q[9];
ry(0.8416841533646019) q[8];
ry(-0.07168704989057684) q[9];
cx q[8],q[9];
ry(0.5298925504364775) q[9];
ry(2.6146216824202586) q[10];
cx q[9],q[10];
ry(0.00016838743625907616) q[9];
ry(-0.004665850514824044) q[10];
cx q[9],q[10];
ry(-2.6049880271836603) q[10];
ry(-0.37827302476800817) q[11];
cx q[10],q[11];
ry(-3.126757898230146) q[10];
ry(2.398949209263251) q[11];
cx q[10],q[11];
ry(-1.633597274261805) q[11];
ry(-2.257231510357723) q[12];
cx q[11],q[12];
ry(1.0339505482225784) q[11];
ry(-3.0715060534596157) q[12];
cx q[11],q[12];
ry(0.36007212271918043) q[12];
ry(1.6291477359979922) q[13];
cx q[12],q[13];
ry(-2.822559658862054) q[12];
ry(-3.140846556033145) q[13];
cx q[12],q[13];
ry(3.100190807697101) q[13];
ry(-0.39749629790163965) q[14];
cx q[13],q[14];
ry(-0.1165518842453457) q[13];
ry(-3.0395552757566575) q[14];
cx q[13],q[14];
ry(-1.774502156141302) q[14];
ry(-1.017408259624057) q[15];
cx q[14],q[15];
ry(2.1203962815105983) q[14];
ry(-0.00982392013876251) q[15];
cx q[14],q[15];
ry(-1.4961598658128379) q[15];
ry(0.9285741327110326) q[16];
cx q[15],q[16];
ry(0.0014787553942197105) q[15];
ry(2.184968352622914) q[16];
cx q[15],q[16];
ry(0.691494761483972) q[16];
ry(-1.233559037462086) q[17];
cx q[16],q[17];
ry(0.11450644530005559) q[16];
ry(-1.472262673970314) q[17];
cx q[16],q[17];
ry(0.813617693253958) q[17];
ry(2.2753672841780475) q[18];
cx q[17],q[18];
ry(0.4632968883062052) q[17];
ry(3.1413416937205034) q[18];
cx q[17],q[18];
ry(-2.0214610547239618) q[18];
ry(-1.203146988332507) q[19];
cx q[18],q[19];
ry(0.2116312539418876) q[18];
ry(-2.6423159715944036) q[19];
cx q[18],q[19];
ry(2.8628504918636892) q[0];
ry(-1.2482939686902723) q[1];
cx q[0],q[1];
ry(0.06846181514830452) q[0];
ry(-0.5868236726565166) q[1];
cx q[0],q[1];
ry(1.7046568234623631) q[1];
ry(-1.3649414947122658) q[2];
cx q[1],q[2];
ry(2.1045707225071766) q[1];
ry(2.2299234637516756) q[2];
cx q[1],q[2];
ry(-2.4923436894242488) q[2];
ry(-0.4998537138885464) q[3];
cx q[2],q[3];
ry(2.299424141531226) q[2];
ry(-2.193292773362049) q[3];
cx q[2],q[3];
ry(-1.3343279468623683) q[3];
ry(-2.0093614152484163) q[4];
cx q[3],q[4];
ry(-3.1411676557288195) q[3];
ry(0.2926947516145354) q[4];
cx q[3],q[4];
ry(-0.0726844905145498) q[4];
ry(-2.5661612923634403) q[5];
cx q[4],q[5];
ry(-0.20728854912119055) q[4];
ry(-3.141584330927954) q[5];
cx q[4],q[5];
ry(-3.0750966836889444) q[5];
ry(-0.8157897604210913) q[6];
cx q[5],q[6];
ry(-0.35616837598416495) q[5];
ry(-1.7051110231009128) q[6];
cx q[5],q[6];
ry(2.583591539805569) q[6];
ry(-1.0367961485112476) q[7];
cx q[6],q[7];
ry(0.006150515871464535) q[6];
ry(-3.140861584530345) q[7];
cx q[6],q[7];
ry(-2.105679425231421) q[7];
ry(1.1209659429853946) q[8];
cx q[7],q[8];
ry(0.0666426450487672) q[7];
ry(2.2976535007165966) q[8];
cx q[7],q[8];
ry(-1.4061038517383175) q[8];
ry(-0.9132984700285727) q[9];
cx q[8],q[9];
ry(2.052471730341396) q[8];
ry(2.916007219491717) q[9];
cx q[8],q[9];
ry(2.1174052278180624) q[9];
ry(-2.624263742343522) q[10];
cx q[9],q[10];
ry(3.1333592016058103) q[9];
ry(2.9103963679238545) q[10];
cx q[9],q[10];
ry(-2.646430050336503) q[10];
ry(2.242670388152648) q[11];
cx q[10],q[11];
ry(0.13976616405202932) q[10];
ry(-2.7576608492459376) q[11];
cx q[10],q[11];
ry(-0.4445101038349488) q[11];
ry(-0.9643334586506637) q[12];
cx q[11],q[12];
ry(-3.0965405340819028) q[11];
ry(0.5356065984272692) q[12];
cx q[11],q[12];
ry(1.8627502129597593) q[12];
ry(2.8098794622721974) q[13];
cx q[12],q[13];
ry(-1.2314453017869544) q[12];
ry(-1.715133863762266) q[13];
cx q[12],q[13];
ry(2.267701908522643) q[13];
ry(-1.516463315059963) q[14];
cx q[13],q[14];
ry(3.135035530131336) q[13];
ry(-2.9740663354927053) q[14];
cx q[13],q[14];
ry(-1.0304946865831386) q[14];
ry(1.603440911073668) q[15];
cx q[14],q[15];
ry(2.452992318436381) q[14];
ry(-3.1242395950348896) q[15];
cx q[14],q[15];
ry(-0.8720520303199136) q[15];
ry(-0.9949237505043776) q[16];
cx q[15],q[16];
ry(-3.141115663601122) q[15];
ry(3.138929888793959) q[16];
cx q[15],q[16];
ry(2.1445430598261073) q[16];
ry(0.6915876545167147) q[17];
cx q[16],q[17];
ry(2.9013292564195416) q[16];
ry(-1.6599251414911282) q[17];
cx q[16],q[17];
ry(-0.6163876259299758) q[17];
ry(-2.2239375574979166) q[18];
cx q[17],q[18];
ry(1.513615558152071) q[17];
ry(0.8352186831312425) q[18];
cx q[17],q[18];
ry(-2.251312909819582) q[18];
ry(-0.29154685717580886) q[19];
cx q[18],q[19];
ry(-0.6494826686644027) q[18];
ry(-0.032065655038633845) q[19];
cx q[18],q[19];
ry(0.9777499733742475) q[0];
ry(0.020796703569595287) q[1];
cx q[0],q[1];
ry(0.08414187715972776) q[0];
ry(-0.0920759870048822) q[1];
cx q[0],q[1];
ry(3.0931702435944324) q[1];
ry(0.3546237420238141) q[2];
cx q[1],q[2];
ry(0.09513305943269962) q[1];
ry(0.8406231661165824) q[2];
cx q[1],q[2];
ry(0.33441383805082187) q[2];
ry(0.8844355633318332) q[3];
cx q[2],q[3];
ry(-2.1109865351029478) q[2];
ry(-2.3053291884613047) q[3];
cx q[2],q[3];
ry(0.7042881421445703) q[3];
ry(2.3731489735718876) q[4];
cx q[3],q[4];
ry(2.710303499490772) q[3];
ry(-2.344130156402231) q[4];
cx q[3],q[4];
ry(1.64405351570315) q[4];
ry(1.4911496859913838) q[5];
cx q[4],q[5];
ry(-3.1386032746707357) q[4];
ry(0.08092707081288797) q[5];
cx q[4],q[5];
ry(2.494396994756217) q[5];
ry(2.225590289002818) q[6];
cx q[5],q[6];
ry(2.5683930036047395) q[5];
ry(-3.1086094644249145) q[6];
cx q[5],q[6];
ry(-1.6244613275181221) q[6];
ry(-1.1730394343108737) q[7];
cx q[6],q[7];
ry(0.027040212698125683) q[6];
ry(2.681437696921755) q[7];
cx q[6],q[7];
ry(2.156122242678941) q[7];
ry(-1.8480922387480407) q[8];
cx q[7],q[8];
ry(3.0734624300560296) q[7];
ry(-0.001283957173684236) q[8];
cx q[7],q[8];
ry(1.0510725441759208) q[8];
ry(-1.6210023662592894) q[9];
cx q[8],q[9];
ry(-1.429632620813343) q[8];
ry(-0.07980928645483318) q[9];
cx q[8],q[9];
ry(-1.6662897392886773) q[9];
ry(-1.2945825773901625) q[10];
cx q[9],q[10];
ry(3.139350633726312) q[9];
ry(-3.114170168667227) q[10];
cx q[9],q[10];
ry(1.903766700407698) q[10];
ry(2.427751204664101) q[11];
cx q[10],q[11];
ry(-3.116354002122403) q[10];
ry(0.8235165658048418) q[11];
cx q[10],q[11];
ry(0.8876457555977942) q[11];
ry(-1.9486481446487547) q[12];
cx q[11],q[12];
ry(0.021700734447653903) q[11];
ry(0.0011559977448447611) q[12];
cx q[11],q[12];
ry(-0.8545780326977919) q[12];
ry(2.39980387881832) q[13];
cx q[12],q[13];
ry(-1.3409319932160297) q[12];
ry(-0.3885221751668167) q[13];
cx q[12],q[13];
ry(-2.454377027439844) q[13];
ry(1.7508506088201266) q[14];
cx q[13],q[14];
ry(-3.0367611514589345) q[13];
ry(3.0975865050829667) q[14];
cx q[13],q[14];
ry(1.4094655619874663) q[14];
ry(2.251407538756813) q[15];
cx q[14],q[15];
ry(-0.4330179330917199) q[14];
ry(0.8233948983635466) q[15];
cx q[14],q[15];
ry(0.17971164028081724) q[15];
ry(-2.7747453234974455) q[16];
cx q[15],q[16];
ry(3.1409973579435753) q[15];
ry(-0.17935896058785517) q[16];
cx q[15],q[16];
ry(0.21533042725817264) q[16];
ry(-2.0633041209863467) q[17];
cx q[16],q[17];
ry(1.0170728243598586) q[16];
ry(0.5874691124929222) q[17];
cx q[16],q[17];
ry(-1.297826154210588) q[17];
ry(-2.6634138794341613) q[18];
cx q[17],q[18];
ry(1.1836657842800662) q[17];
ry(-2.1816920501833352) q[18];
cx q[17],q[18];
ry(1.0224700924550119) q[18];
ry(2.9280779789648963) q[19];
cx q[18],q[19];
ry(-2.3261267810652435) q[18];
ry(-3.0355169663531187) q[19];
cx q[18],q[19];
ry(3.1049804752150876) q[0];
ry(1.3746026305915242) q[1];
cx q[0],q[1];
ry(-0.3807282109705952) q[0];
ry(0.25257209552843973) q[1];
cx q[0],q[1];
ry(-3.0486002401584753) q[1];
ry(-0.8848445349879088) q[2];
cx q[1],q[2];
ry(2.517557076129331) q[1];
ry(0.333014029867096) q[2];
cx q[1],q[2];
ry(0.1449141447603104) q[2];
ry(0.03740732860080698) q[3];
cx q[2],q[3];
ry(-2.7726702975453272) q[2];
ry(-2.9966985125196937) q[3];
cx q[2],q[3];
ry(2.830082267927537) q[3];
ry(3.073478535673898) q[4];
cx q[3],q[4];
ry(0.0024862327179278054) q[3];
ry(-3.1337633810767995) q[4];
cx q[3],q[4];
ry(-2.869782563242192) q[4];
ry(-0.6571900077571078) q[5];
cx q[4],q[5];
ry(3.141335680449063) q[4];
ry(3.121146322422687) q[5];
cx q[4],q[5];
ry(-1.9733814927725257) q[5];
ry(1.3864417892118865) q[6];
cx q[5],q[6];
ry(2.37739948855632) q[5];
ry(3.0850599564053156) q[6];
cx q[5],q[6];
ry(-2.5822868045595033) q[6];
ry(-2.9507081401089827) q[7];
cx q[6],q[7];
ry(-0.46452262161494823) q[6];
ry(-1.9816904311166508) q[7];
cx q[6],q[7];
ry(-1.7092742299357093) q[7];
ry(0.6003875303248636) q[8];
cx q[7],q[8];
ry(-0.016824364839947325) q[7];
ry(-0.019929059867489052) q[8];
cx q[7],q[8];
ry(-2.7418888071047585) q[8];
ry(1.7363205204976055) q[9];
cx q[8],q[9];
ry(-1.5109065162804036) q[8];
ry(-0.5484517890790506) q[9];
cx q[8],q[9];
ry(-1.7132697530925576) q[9];
ry(0.021547603630546072) q[10];
cx q[9],q[10];
ry(3.137077461594224) q[9];
ry(-0.07069114585671432) q[10];
cx q[9],q[10];
ry(-1.1978581121846839) q[10];
ry(2.2796061822423757) q[11];
cx q[10],q[11];
ry(0.3349327085498395) q[10];
ry(-0.0800249520877896) q[11];
cx q[10],q[11];
ry(0.46337495652791133) q[11];
ry(0.21958281079485967) q[12];
cx q[11],q[12];
ry(0.00017002009032562881) q[11];
ry(-3.1370713912764074) q[12];
cx q[11],q[12];
ry(-0.38060631608736806) q[12];
ry(-0.4242820966266754) q[13];
cx q[12],q[13];
ry(2.922708855623299) q[12];
ry(-0.6887166195010376) q[13];
cx q[12],q[13];
ry(-0.2261706522379603) q[13];
ry(-0.9169694696022633) q[14];
cx q[13],q[14];
ry(3.114619689828365) q[13];
ry(-0.04423258022637154) q[14];
cx q[13],q[14];
ry(-1.1800291132953493) q[14];
ry(2.7332452777400533) q[15];
cx q[14],q[15];
ry(-0.7581255887004864) q[14];
ry(-1.909589939971148) q[15];
cx q[14],q[15];
ry(-2.727490761120499) q[15];
ry(1.5632900245535) q[16];
cx q[15],q[16];
ry(-3.120644029490105) q[15];
ry(3.0301162323767046) q[16];
cx q[15],q[16];
ry(0.39921742097945945) q[16];
ry(-1.5346529348232132) q[17];
cx q[16],q[17];
ry(2.3291811479125384) q[16];
ry(-0.010258544715043998) q[17];
cx q[16],q[17];
ry(1.6130834237771343) q[17];
ry(2.9354615335948924) q[18];
cx q[17],q[18];
ry(-0.052049082416475746) q[17];
ry(-1.7400770626446391) q[18];
cx q[17],q[18];
ry(-0.06932890521335366) q[18];
ry(-1.706976948961864) q[19];
cx q[18],q[19];
ry(-0.8911084592109768) q[18];
ry(-2.5312646307053015) q[19];
cx q[18],q[19];
ry(-2.914439231952064) q[0];
ry(-1.2162717891775747) q[1];
ry(1.759447057980562) q[2];
ry(1.3178595986584591) q[3];
ry(0.25731260859961025) q[4];
ry(2.801290632277695) q[5];
ry(-0.5994982377822824) q[6];
ry(0.6316759214736066) q[7];
ry(2.5595994990309405) q[8];
ry(-0.02030788119921287) q[9];
ry(-2.12118614415376) q[10];
ry(1.1850891386899312) q[11];
ry(2.635703995904772) q[12];
ry(0.21133941854685823) q[13];
ry(0.5064931395894776) q[14];
ry(-0.0044091932601793005) q[15];
ry(-1.9852062623311533) q[16];
ry(-0.03656669396591011) q[17];
ry(-2.992873676579398) q[18];
ry(2.062325458451226) q[19];