OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(2.6328859921245) q[0];
rz(2.2743760923114467) q[0];
ry(2.964206188024009) q[1];
rz(2.7347579486182165) q[1];
ry(2.033138269147795) q[2];
rz(-0.15192406989234097) q[2];
ry(1.3813942588456616) q[3];
rz(-2.651928200892291) q[3];
ry(-3.1415822212804994) q[4];
rz(1.5973686855381377) q[4];
ry(-3.1405273293791596) q[5];
rz(-0.9266051766595531) q[5];
ry(3.140275111197739) q[6];
rz(-0.4043298267806171) q[6];
ry(3.141091159686001) q[7];
rz(1.625304901319127) q[7];
ry(-3.1300999352183827) q[8];
rz(-2.7746373866523766) q[8];
ry(-2.9822936368396356) q[9];
rz(0.6295299915955833) q[9];
ry(2.635103821128596) q[10];
rz(0.9607357867144461) q[10];
ry(-2.3230260222571233) q[11];
rz(-1.0914207712638595) q[11];
ry(1.5698424006147684) q[12];
rz(3.1397232129323203) q[12];
ry(1.5740149715226994) q[13];
rz(2.356569678708288) q[13];
ry(0.3345842883432715) q[14];
rz(-3.1377284053702423) q[14];
ry(2.7569573095829085) q[15];
rz(0.53905114405502) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-2.7180362082061733) q[0];
rz(-0.4381349361431855) q[0];
ry(-2.158871823701582) q[1];
rz(-2.5581453523204063) q[1];
ry(-1.437324837247998) q[2];
rz(3.073596086191153) q[2];
ry(-0.4140751855471869) q[3];
rz(3.034235379534675) q[3];
ry(-1.2935442693977368) q[4];
rz(2.5903409092888707) q[4];
ry(-2.7646697215159595) q[5];
rz(2.4230269438013097) q[5];
ry(-0.09632644096313657) q[6];
rz(2.6662082559019975) q[6];
ry(-3.1404880618482935) q[7];
rz(0.5530968056456577) q[7];
ry(-3.135444946246021) q[8];
rz(2.186997969554853) q[8];
ry(-2.947143141109715) q[9];
rz(2.231548426632292) q[9];
ry(2.7966883592731206) q[10];
rz(-1.732000432010973) q[10];
ry(-2.8514176201138803) q[11];
rz(0.7409827613213796) q[11];
ry(-0.017523789493340658) q[12];
rz(2.695706798625111) q[12];
ry(-3.1355275849154784) q[13];
rz(1.8904188275405718) q[13];
ry(1.464554065103072) q[14];
rz(-2.8597991578302078) q[14];
ry(2.2719121423629387) q[15];
rz(1.7320588176991387) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-3.0774431420199684) q[0];
rz(1.6250318651142142) q[0];
ry(0.2789785142418658) q[1];
rz(1.2406819542412872) q[1];
ry(-0.0001446461187173398) q[2];
rz(0.919066326428224) q[2];
ry(0.00010016979725691044) q[3];
rz(1.1184032655898712) q[3];
ry(0.0003286940807942204) q[4];
rz(-1.0507937701338115) q[4];
ry(-0.00014677487377401377) q[5];
rz(1.5583491431813583) q[5];
ry(-3.141533409725796) q[6];
rz(2.7058617968836938) q[6];
ry(-6.0121330913709414e-05) q[7];
rz(-2.865619674181913) q[7];
ry(3.135485898294136) q[8];
rz(1.5875037790113986) q[8];
ry(-3.135708652644551) q[9];
rz(-3.0062894621817042) q[9];
ry(1.5847493324227315) q[10];
rz(-0.3429432472949032) q[10];
ry(-0.6254801812647441) q[11];
rz(-3.1151341150470375) q[11];
ry(0.005471790226165692) q[12];
rz(2.2603247158963633) q[12];
ry(0.00637274243918462) q[13];
rz(0.9149477081253813) q[13];
ry(0.38360582462034504) q[14];
rz(-1.8708705195266138) q[14];
ry(-3.123144177758137) q[15];
rz(2.9351832734053875) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-2.1336229649148546) q[0];
rz(1.5449895659214095) q[0];
ry(0.401599117706537) q[1];
rz(-2.511481016906755) q[1];
ry(0.18661258225783772) q[2];
rz(1.6533650500827195) q[2];
ry(-2.979631248036499) q[3];
rz(-1.0430719001652466) q[3];
ry(1.4592607753984241) q[4];
rz(-1.3507203534137961) q[4];
ry(-1.8305114094604489) q[5];
rz(0.2382148998206839) q[5];
ry(-3.0483521415310793) q[6];
rz(-0.40728306530327885) q[6];
ry(-3.1397731358735363) q[7];
rz(-1.4043335377762989) q[7];
ry(-1.5614470508532559) q[8];
rz(-2.596555506666729) q[8];
ry(1.9066637804503834) q[9];
rz(3.025104331346622) q[9];
ry(-2.281470425535157) q[10];
rz(0.23517829147484726) q[10];
ry(-0.09290533715521665) q[11];
rz(2.338696780043619) q[11];
ry(-0.0064881755395569726) q[12];
rz(1.3316650493223126) q[12];
ry(3.1195747701579135) q[13];
rz(0.4497236308028328) q[13];
ry(-1.2672860475002072) q[14];
rz(1.7748486634917573) q[14];
ry(-2.431066129818943) q[15];
rz(-0.6841565337032742) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-3.017962677109219) q[0];
rz(0.6255558322959183) q[0];
ry(-1.3219755537503177) q[1];
rz(1.2852514750380495) q[1];
ry(-3.141459741032768) q[2];
rz(0.9371153224682159) q[2];
ry(3.141574834972538) q[3];
rz(-0.012460065976056534) q[3];
ry(0.010761905706778487) q[4];
rz(-0.6292687217782653) q[4];
ry(3.130681080059025) q[5];
rz(-1.609253000735809) q[5];
ry(-1.5708420751225818) q[6];
rz(0.0009710659227682257) q[6];
ry(-1.570736837791476) q[7];
rz(-3.1404351512771624) q[7];
ry(3.118007573550669) q[8];
rz(2.052589542477094) q[8];
ry(-0.011966372448535267) q[9];
rz(2.1371386320619576) q[9];
ry(-0.003380863066559978) q[10];
rz(1.981068254912139) q[10];
ry(0.006338291847431954) q[11];
rz(1.776321129422236) q[11];
ry(1.5733616972060838) q[12];
rz(-0.13664379765863635) q[12];
ry(1.5722955061170976) q[13];
rz(3.001105030528147) q[13];
ry(-0.5786471343105797) q[14];
rz(-1.7572382983943236) q[14];
ry(0.015690342378963044) q[15];
rz(0.18501556611173606) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(1.701959620704095) q[0];
rz(-2.377582638093946) q[0];
ry(-1.2351355569536766) q[1];
rz(-2.076031497753977) q[1];
ry(-1.315899382293125) q[2];
rz(-1.6363274097925682) q[2];
ry(2.002987942684712) q[3];
rz(0.8528267546269149) q[3];
ry(-3.9871178135975065e-05) q[4];
rz(-1.5677097997720206) q[4];
ry(2.977752682201642) q[5];
rz(-2.693182203105627) q[5];
ry(-1.8880911504612978) q[6];
rz(-0.10132967020655224) q[6];
ry(-1.8880448469888593) q[7];
rz(-1.5606106306128384) q[7];
ry(-0.11340516178775317) q[8];
rz(-1.5979002176864598) q[8];
ry(-1.5376151922668937) q[9];
rz(3.043845634092917) q[9];
ry(-1.671508632771613) q[10];
rz(2.789160078587759) q[10];
ry(2.808714522199281) q[11];
rz(0.8960300728662656) q[11];
ry(-1.5788321150056601) q[12];
rz(-2.4768924077983274) q[12];
ry(1.5595317006702736) q[13];
rz(1.8830080912610931) q[13];
ry(3.1345468325212575) q[14];
rz(1.8716135638944733) q[14];
ry(0.48236271771841466) q[15];
rz(1.5428942918020683) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(1.9605817325761128) q[0];
rz(-1.8618205559131975) q[0];
ry(0.6488308045893643) q[1];
rz(-0.5909373846414594) q[1];
ry(1.5706670775504519) q[2];
rz(-0.0008547906542260536) q[2];
ry(1.5702200024477957) q[3];
rz(-0.0005683633781208286) q[3];
ry(0.03586580909986137) q[4];
rz(-2.3144180533491547) q[4];
ry(3.1354550592130424) q[5];
rz(-1.049893121110986) q[5];
ry(3.118037404295726) q[6];
rz(-2.454384237036773) q[6];
ry(-0.18302215284914186) q[7];
rz(-0.28570804225978463) q[7];
ry(-1.8338566388740416) q[8];
rz(0.7371001705074578) q[8];
ry(2.285401700085101) q[9];
rz(0.2625428027721833) q[9];
ry(-0.18134834033782368) q[10];
rz(1.1475563930620405) q[10];
ry(-0.9519504514580523) q[11];
rz(0.8330249856174322) q[11];
ry(0.00047171726808681314) q[12];
rz(-2.0324325102452243) q[12];
ry(0.00026025695317244413) q[13];
rz(-1.1796025886491686) q[13];
ry(2.944310776947059) q[14];
rz(1.828842628874538) q[14];
ry(-1.4507981949559268) q[15];
rz(2.195690948584293) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(1.3394223223492867) q[0];
rz(-0.21817789747455274) q[0];
ry(-2.967109680024243) q[1];
rz(-0.7778100188212707) q[1];
ry(-1.5708757470068193) q[2];
rz(1.7314626858319793) q[2];
ry(-1.571069110964302) q[3];
rz(-1.4706189711818274) q[3];
ry(3.1415692264592567) q[4];
rz(-2.404086506325105) q[4];
ry(-3.141581234205668) q[5];
rz(1.1582145088005549) q[5];
ry(-3.141426443455321) q[6];
rz(2.2085167725627133) q[6];
ry(2.798628415945359e-05) q[7];
rz(-2.2629203602582737) q[7];
ry(3.1352717651073294) q[8];
rz(0.5318947415539973) q[8];
ry(-0.0021459782586639164) q[9];
rz(0.46455104311216644) q[9];
ry(-3.124542942778018) q[10];
rz(-1.664928530719702) q[10];
ry(-0.021530371719320283) q[11];
rz(2.9426390635144597) q[11];
ry(-0.0024557576162663845) q[12];
rz(-2.2306870872450113) q[12];
ry(-3.139498964343402) q[13];
rz(-2.1585320156209153) q[13];
ry(-1.8612579289877405) q[14];
rz(-0.5862865543532579) q[14];
ry(-1.7643570153308803) q[15];
rz(-2.9456973976371748) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(1.5382531330315716) q[0];
rz(-1.7751637756556673) q[0];
ry(-2.1663920012469533) q[1];
rz(-1.324704799388665) q[1];
ry(1.0146955843387282) q[2];
rz(2.712736013633183) q[2];
ry(-3.073251623536072) q[3];
rz(-0.5372058774093292) q[3];
ry(1.9060453136824798) q[4];
rz(-2.419909289775284) q[4];
ry(2.096522302807303) q[5];
rz(-1.1055740054190777) q[5];
ry(-2.99451541871291) q[6];
rz(-1.7830085278945313) q[6];
ry(3.1391504108245685) q[7];
rz(3.100318029198463) q[7];
ry(2.767293248429944) q[8];
rz(1.3140480158518981) q[8];
ry(-1.0266117261529764) q[9];
rz(-0.5163124752262903) q[9];
ry(-1.6553442041788504) q[10];
rz(-2.2813045220170967) q[10];
ry(2.4615037839437033) q[11];
rz(-3.014082995982205) q[11];
ry(-3.140085129811514) q[12];
rz(1.3989002974910116) q[12];
ry(3.1336077847745676) q[13];
rz(2.1706030330574237) q[13];
ry(1.0700328598016258) q[14];
rz(2.780097380711138) q[14];
ry(2.2147850500152533) q[15];
rz(-1.7162520909104577) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(3.0947038989230062) q[0];
rz(-0.9934974820712483) q[0];
ry(0.22212195303925952) q[1];
rz(-0.45258257358866594) q[1];
ry(2.207506990355589) q[2];
rz(2.9196788760560133) q[2];
ry(1.145764144048078) q[3];
rz(0.5679170976388571) q[3];
ry(0.007179901677503153) q[4];
rz(-0.8633253322554095) q[4];
ry(-0.0071235127402823695) q[5];
rz(2.9101126982575027) q[5];
ry(3.13519541709824) q[6];
rz(3.104207121471424) q[6];
ry(3.141376677683991) q[7];
rz(-2.207644115739066) q[7];
ry(1.5636851797654345) q[8];
rz(-0.7461732118966399) q[8];
ry(1.5790816305423867) q[9];
rz(2.7375208900231556) q[9];
ry(-2.958790278743245) q[10];
rz(-0.27954325732314733) q[10];
ry(-0.9196242474217025) q[11];
rz(-1.5139765849496352) q[11];
ry(-3.136806699267965) q[12];
rz(0.293663976249408) q[12];
ry(-1.0553922504819015e-05) q[13];
rz(2.8302724606845135) q[13];
ry(2.9735667152846745) q[14];
rz(-1.9627992194291404) q[14];
ry(0.00809247882664787) q[15];
rz(0.7699483932784158) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(0.9187175921216552) q[0];
rz(0.0618602891849713) q[0];
ry(-1.570459199799182) q[1];
rz(0.42094014356898685) q[1];
ry(-1.2509191007121523) q[2];
rz(3.0222544756970313) q[2];
ry(1.2902708575942743) q[3];
rz(0.32701318362103987) q[3];
ry(-3.109215939203454) q[4];
rz(-3.0801448270059217) q[4];
ry(-3.1402418218955406) q[5];
rz(-0.6419316282162768) q[5];
ry(1.5753754510693154) q[6];
rz(-1.0396286652125344) q[6];
ry(1.5904092555674678) q[7];
rz(-0.07181596462458463) q[7];
ry(2.4198457323538305) q[8];
rz(-2.336329968160562) q[8];
ry(-2.526781633894118) q[9];
rz(-1.2243768851450803) q[9];
ry(-2.4780179988223794) q[10];
rz(1.4800286212068623) q[10];
ry(0.743014067199148) q[11];
rz(1.9818745019055046) q[11];
ry(-1.5718323217207428) q[12];
rz(2.638901925025659) q[12];
ry(-1.5692179776883588) q[13];
rz(2.0163634528152596) q[13];
ry(-1.2391797269935456) q[14];
rz(-1.0997618645288016) q[14];
ry(-0.0604104956355238) q[15];
rz(-1.4932427948729208) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-2.1862277503290155) q[0];
rz(1.8200514278606288) q[0];
ry(1.0081540470912218) q[1];
rz(-2.0200019300741796) q[1];
ry(-2.453796603389651) q[2];
rz(-2.445602217264176) q[2];
ry(-1.546063731772188) q[3];
rz(-0.20412070657863654) q[3];
ry(-0.013273214948391792) q[4];
rz(3.1047593858313833) q[4];
ry(-3.090622776715559) q[5];
rz(2.1748630015501753) q[5];
ry(-3.1408184449266963) q[6];
rz(2.649957430355765) q[6];
ry(-3.1404694436169582) q[7];
rz(-2.1736468190381544) q[7];
ry(1.5254812625458918) q[8];
rz(-2.0430493349729346) q[8];
ry(-0.7344291095974671) q[9];
rz(2.2388869025698495) q[9];
ry(1.8531583361712949) q[10];
rz(-2.017618975114812) q[10];
ry(-2.7319755212067833) q[11];
rz(1.1694053891898237) q[11];
ry(3.1400579618159883) q[12];
rz(-1.4324075490376371) q[12];
ry(3.1398933594216394) q[13];
rz(1.0939159875035052) q[13];
ry(1.2857242986146835) q[14];
rz(-2.9230490454718554) q[14];
ry(2.4277711664736064) q[15];
rz(-0.2583413304135859) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(1.4024167188407992) q[0];
rz(1.1225996319325546) q[0];
ry(2.8003554647386735) q[1];
rz(-2.491639763752771) q[1];
ry(0.13425051656400455) q[2];
rz(-0.9888434791313917) q[2];
ry(-0.7162674764369374) q[3];
rz(-1.281068682141854) q[3];
ry(-1.5086168947278824) q[4];
rz(-1.574535073591136) q[4];
ry(-3.1294999980441247) q[5];
rz(-1.2319995012296825) q[5];
ry(-1.5476380169700914) q[6];
rz(-1.0202693167552372) q[6];
ry(1.5473247181350713) q[7];
rz(-0.7458360668227488) q[7];
ry(-3.031538054355179) q[8];
rz(-1.2812017869105) q[8];
ry(1.4064045021168141) q[9];
rz(0.49118632251788963) q[9];
ry(-2.3403518064396414) q[10];
rz(1.2520969436386449) q[10];
ry(0.30640075896382335) q[11];
rz(0.5109586189205632) q[11];
ry(1.5797304740473288) q[12];
rz(1.7143780393765229) q[12];
ry(1.558896480620322) q[13];
rz(-0.15827641380222698) q[13];
ry(-1.1706842846911998) q[14];
rz(1.3095910277308305) q[14];
ry(-0.8422100023875434) q[15];
rz(-0.8011641871117497) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-0.6821857743049219) q[0];
rz(2.8254130002587288) q[0];
ry(-2.4348856423489265) q[1];
rz(-1.3354205536139547) q[1];
ry(3.1307832299383267) q[2];
rz(0.8382354946685444) q[2];
ry(-0.006566681009869808) q[3];
rz(-2.4132508181729273) q[3];
ry(-1.5719288329116825) q[4];
rz(3.136867291046645) q[4];
ry(-1.566289047025643) q[5];
rz(-0.004109199062805935) q[5];
ry(-0.08573456734870888) q[6];
rz(-2.220545708106368) q[6];
ry(2.906958183490565) q[7];
rz(0.845098271066952) q[7];
ry(-2.4437247450840434) q[8];
rz(-2.6143349002220075) q[8];
ry(-1.330341759096904) q[9];
rz(-0.48047876819397667) q[9];
ry(1.3096578885336376) q[10];
rz(3.0872505400476755) q[10];
ry(3.0025333876067717) q[11];
rz(2.2818492647458473) q[11];
ry(0.0004254313915429264) q[12];
rz(-1.992772747023019) q[12];
ry(3.140142809030887) q[13];
rz(2.513109697919618) q[13];
ry(3.0353957542780066) q[14];
rz(2.4408406051619815) q[14];
ry(0.1846705799378873) q[15];
rz(-1.5022609026111928) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-2.3807364952697565) q[0];
rz(2.160631095681568) q[0];
ry(-2.43567786614988) q[1];
rz(1.7555842168276101) q[1];
ry(-0.801652568472532) q[2];
rz(1.5825312757194552) q[2];
ry(-1.7615042247850257) q[3];
rz(1.359650606528147) q[3];
ry(1.5751963369889224) q[4];
rz(1.5324377053549068) q[4];
ry(-1.5662608879116764) q[5];
rz(0.3173749475652703) q[5];
ry(-0.001964132837463409) q[6];
rz(-1.8565170633400556) q[6];
ry(-0.016683729854655282) q[7];
rz(1.4316025886370758) q[7];
ry(-0.002007373947454205) q[8];
rz(3.0144836545592266) q[8];
ry(-3.1409374568888953) q[9];
rz(-2.0560257152505255) q[9];
ry(2.607097486762354) q[10];
rz(3.06180436460517) q[10];
ry(-0.5798616185698776) q[11];
rz(0.2686096235060496) q[11];
ry(3.1334348371150793) q[12];
rz(0.6594324241328557) q[12];
ry(0.018706827129578878) q[13];
rz(0.6679992196840248) q[13];
ry(-2.8937721294616763) q[14];
rz(2.3275842246089593) q[14];
ry(1.7489446159618234) q[15];
rz(0.7278816527742747) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-0.5441163608823846) q[0];
rz(-1.5816864140538462) q[0];
ry(-0.5725186213871734) q[1];
rz(-2.2396846294402577) q[1];
ry(1.3667942336556616) q[2];
rz(-2.9572881049859228) q[2];
ry(-2.9200033400046217) q[3];
rz(-2.168733103829168) q[3];
ry(-0.0007261209457210618) q[4];
rz(-0.9555639303546609) q[4];
ry(0.0004436188359337563) q[5];
rz(-1.2587556787447542) q[5];
ry(-1.5695903830366618) q[6];
rz(-1.4271782859024364) q[6];
ry(-1.5665602697657306) q[7];
rz(1.4916557661516043) q[7];
ry(-2.3476750744372303) q[8];
rz(-0.8374712284898073) q[8];
ry(3.082526817368037) q[9];
rz(2.404208968771933) q[9];
ry(2.6326931597109544) q[10];
rz(-2.2079855950843825) q[10];
ry(-2.875849863140338) q[11];
rz(1.1475806073145325) q[11];
ry(0.0009441243644103043) q[12];
rz(0.6250383322661923) q[12];
ry(3.141294901220705) q[13];
rz(1.7751417697381007) q[13];
ry(1.3499062236295822) q[14];
rz(2.76214077093386) q[14];
ry(-1.4505222038934986) q[15];
rz(-3.02459847981756) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(2.599563105003785) q[0];
rz(2.1676538595082544) q[0];
ry(1.0018484623637127) q[1];
rz(1.7920390817641938) q[1];
ry(0.6818486355534303) q[2];
rz(-0.9843368946291571) q[2];
ry(-0.8727656924428784) q[3];
rz(2.123473095328815) q[3];
ry(2.184377294417977) q[4];
rz(0.18700106936269908) q[4];
ry(2.2195679893686764) q[5];
rz(-2.2633208358738943) q[5];
ry(-0.024953383752086868) q[6];
rz(2.9970973140023673) q[6];
ry(3.1116438104247863) q[7];
rz(3.052679964828296) q[7];
ry(-1.5713839910007519) q[8];
rz(-3.141436851849517) q[8];
ry(-1.5710033988826098) q[9];
rz(0.00017441645317773936) q[9];
ry(2.438172854527163) q[10];
rz(-2.7700620099400908) q[10];
ry(2.9082629896267793) q[11];
rz(-2.577853264386075) q[11];
ry(0.6653562821520745) q[12];
rz(3.1403777356545413) q[12];
ry(0.6644018495053998) q[13];
rz(3.130985434226415) q[13];
ry(0.3963256002313509) q[14];
rz(-2.5064447315497556) q[14];
ry(1.5318778689518684) q[15];
rz(1.3720659604302774) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-1.6387906771259235) q[0];
rz(2.0683612900413966) q[0];
ry(-2.004491450246273) q[1];
rz(2.0899518154433316) q[1];
ry(3.141444893099061) q[2];
rz(-2.3233183460069244) q[2];
ry(0.0002173526320987662) q[3];
rz(1.2283743531460753) q[3];
ry(-3.141585227496012) q[4];
rz(-2.957436954995956) q[4];
ry(-0.00041395225466178687) q[5];
rz(-0.8956615672063012) q[5];
ry(1.5718903140956213) q[6];
rz(1.1622076969623236) q[6];
ry(-1.5699972139109477) q[7];
rz(-1.5696023926745353) q[7];
ry(-1.5720411896281936) q[8];
rz(1.5450260748699725) q[8];
ry(-1.5708561481632166) q[9];
rz(1.6217684833518542) q[9];
ry(-1.3923238834414855) q[10];
rz(-0.32821845252161547) q[10];
ry(-1.210610809609014) q[11];
rz(2.747105781165837) q[11];
ry(-1.563182809211737) q[12];
rz(0.545060591599789) q[12];
ry(1.5636502121616167) q[13];
rz(-2.9561978298631755) q[13];
ry(0.623104344451559) q[14];
rz(-0.7285704944174887) q[14];
ry(-2.9177186343681805) q[15];
rz(-2.259632841569678) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-2.754053646347852) q[0];
rz(-1.7678518455640446) q[0];
ry(-1.3947650767038047) q[1];
rz(-0.9703767963652323) q[1];
ry(-2.7809543951796085) q[2];
rz(-1.287756975266444) q[2];
ry(-0.5979186986983586) q[3];
rz(0.898206280733711) q[3];
ry(2.6019566815421378) q[4];
rz(1.1337954053446966) q[4];
ry(2.654939213127951) q[5];
rz(1.9324812828120361) q[5];
ry(3.1403638676086207) q[6];
rz(-0.4079831179238642) q[6];
ry(1.5787519403948922) q[7];
rz(2.977435855562506) q[7];
ry(-1.5697998108108524) q[8];
rz(-1.0346595242673882) q[8];
ry(-1.5683567638108766) q[9];
rz(2.173707138993812) q[9];
ry(-2.8486811879233636) q[10];
rz(2.0937763567284273) q[10];
ry(-0.35278284090743384) q[11];
rz(1.4383133326543165) q[11];
ry(-3.1265430479506247) q[12];
rz(0.9528784244644827) q[12];
ry(-0.0009004973118873495) q[13];
rz(-0.31053740691152404) q[13];
ry(-0.49080213078760043) q[14];
rz(1.5015856023575929) q[14];
ry(-3.116808347999477) q[15];
rz(0.7449170928060969) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-0.2780642625054437) q[0];
rz(1.452493075074433) q[0];
ry(-1.0436116913508533) q[1];
rz(1.2338424729805932) q[1];
ry(0.8697632868287002) q[2];
rz(0.6390149545045096) q[2];
ry(0.2918364826687074) q[3];
rz(-0.4759116777137394) q[3];
ry(-3.141463748494651) q[4];
rz(0.06645617042611818) q[4];
ry(3.303479245867982e-05) q[5];
rz(-0.8165521057158727) q[5];
ry(-0.2539308464524899) q[6];
rz(1.5695975959840474) q[6];
ry(0.0005794437060589014) q[7];
rz(1.7349918540580962) q[7];
ry(3.1412904458230524) q[8];
rz(0.5369307119175657) q[8];
ry(-0.0008163114325885346) q[9];
rz(-0.6049993608405849) q[9];
ry(0.587485437893994) q[10];
rz(2.555772003341032) q[10];
ry(2.763398450024466) q[11];
rz(0.8841855537639542) q[11];
ry(-1.5667358371334679) q[12];
rz(1.5641144973328729) q[12];
ry(-1.5615383437811605) q[13];
rz(2.3431145918119944) q[13];
ry(-1.2042270990014075) q[14];
rz(3.097920513371954) q[14];
ry(0.014961138179405253) q[15];
rz(2.6367037169324274) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(7.522607788423842e-05) q[0];
rz(-2.774569869858774) q[0];
ry(3.141425517411914) q[1];
rz(1.496364280512285) q[1];
ry(0.0006387457291365806) q[2];
rz(-0.608229873221056) q[2];
ry(0.000442366271728384) q[3];
rz(-0.5451877218331104) q[3];
ry(-3.1398059070973785) q[4];
rz(2.06134381203998) q[4];
ry(0.0022575372828163177) q[5];
rz(2.0141067209959687) q[5];
ry(-1.5707619700383182) q[6];
rz(2.2331744409286367) q[6];
ry(1.5722161526854004) q[7];
rz(-1.9544189030484584) q[7];
ry(1.6629851671746783) q[8];
rz(2.941538118452968) q[8];
ry(1.8312911260750944) q[9];
rz(1.1431206076974891) q[9];
ry(-2.220045200321215) q[10];
rz(-2.52621523182605) q[10];
ry(2.082857146020399) q[11];
rz(1.9188431885196076) q[11];
ry(1.5609938885363475) q[12];
rz(-3.0693095985386525) q[12];
ry(3.1307871290768166) q[13];
rz(-2.5059471295223865) q[13];
ry(0.1047080966305023) q[14];
rz(1.5067306249578403) q[14];
ry(-3.10607446201555) q[15];
rz(-3.068228597175675) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-1.0300214130121403) q[0];
rz(-0.3456164833166029) q[0];
ry(-0.09661905637363871) q[1];
rz(0.35254457945628204) q[1];
ry(0.7008375975135861) q[2];
rz(-0.023686766040899986) q[2];
ry(-1.7264496198753818) q[3];
rz(2.893483317556793) q[3];
ry(-2.927818638066079) q[4];
rz(2.9364501145211377) q[4];
ry(0.02680419026576253) q[5];
rz(1.762693810294212) q[5];
ry(3.1342928203899687) q[6];
rz(-2.830738427183523) q[6];
ry(3.1072773305741594) q[7];
rz(-2.1603175892461177) q[7];
ry(-0.012672696739406446) q[8];
rz(1.771365423337678) q[8];
ry(0.0009003840476746037) q[9];
rz(-2.7206826679698373) q[9];
ry(-1.5689102820828564) q[10];
rz(-3.141294319812208) q[10];
ry(-1.5685623381357807) q[11];
rz(-0.0012688572470488943) q[11];
ry(0.5567541667533913) q[12];
rz(1.4036084901295145) q[12];
ry(0.026425458633211996) q[13];
rz(-1.9532218112068895) q[13];
ry(2.8737087870476534) q[14];
rz(0.1307342206176954) q[14];
ry(1.5736994535375874) q[15];
rz(-1.661831345813787) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(0.29331622955414005) q[0];
rz(0.6496829907187409) q[0];
ry(-1.5079831166798543) q[1];
rz(1.5867393941461572) q[1];
ry(1.5707506093832624) q[2];
rz(-3.131460511814338) q[2];
ry(-1.5710314027196055) q[3];
rz(1.5251933360269125) q[3];
ry(0.0012609106143293303) q[4];
rz(1.7721966203975308) q[4];
ry(0.0004012991982470261) q[5];
rz(2.9453377361037494) q[5];
ry(-3.1407083822980297) q[6];
rz(2.78889333092393) q[6];
ry(3.141142502810189) q[7];
rz(-0.1967405891527362) q[7];
ry(1.573556974427706) q[8];
rz(-0.03971458126839078) q[8];
ry(1.5713173177277362) q[9];
rz(2.758126824175963) q[9];
ry(1.5704236205211901) q[10];
rz(1.5445255548189654) q[10];
ry(-1.5735942961948775) q[11];
rz(1.6175251198223688) q[11];
ry(-0.0004933793611746304) q[12];
rz(1.018116414861381) q[12];
ry(-3.1411858546315163) q[13];
rz(-0.5279388726062977) q[13];
ry(-0.006258310241778808) q[14];
rz(-0.030892283517530063) q[14];
ry(-2.8637324507809643) q[15];
rz(-0.09890124044654769) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(0.3738307897780318) q[0];
rz(-0.5238046988131374) q[0];
ry(2.150755596916529) q[1];
rz(-0.10977643665500379) q[1];
ry(-0.0021017425577465293) q[2];
rz(-0.021651830278346364) q[2];
ry(2.707666308814639) q[3];
rz(3.0839548991787655) q[3];
ry(-1.5714218360543537) q[4];
rz(-1.5769060088183717) q[4];
ry(1.5713794947624569) q[5];
rz(1.5721308442039739) q[5];
ry(1.5684506236732978) q[6];
rz(0.002127574106185048) q[6];
ry(1.5724764065206775) q[7];
rz(-0.02213262490741445) q[7];
ry(2.4142030090005377) q[8];
rz(-0.11227058533207819) q[8];
ry(0.2595234751486677) q[9];
rz(1.905167983973687) q[9];
ry(-1.963207767425569) q[10];
rz(1.5428431684587904) q[10];
ry(1.4035121540195148) q[11];
rz(-3.057033772396724) q[11];
ry(3.0224936439114343) q[12];
rz(-2.2221837783771035) q[12];
ry(1.4977077294975576) q[13];
rz(1.567309684061404) q[13];
ry(-1.2871719216345667) q[14];
rz(-0.0333196201626182) q[14];
ry(1.6011198227479082) q[15];
rz(1.5749446904104907) q[15];