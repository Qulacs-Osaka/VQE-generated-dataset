OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-2.8842536029562575) q[0];
ry(0.022871129429076653) q[1];
cx q[0],q[1];
ry(-1.8014950604198496) q[0];
ry(0.3480361006682203) q[1];
cx q[0],q[1];
ry(-2.659662314504536) q[1];
ry(-3.1386480832808252) q[2];
cx q[1],q[2];
ry(0.6042638512032832) q[1];
ry(-1.3396072983457161) q[2];
cx q[1],q[2];
ry(0.7593409634737459) q[2];
ry(0.04419339556807582) q[3];
cx q[2],q[3];
ry(-1.6084758157877284) q[2];
ry(1.412355743288214) q[3];
cx q[2],q[3];
ry(-2.7431236382058164) q[3];
ry(0.056957344753836736) q[4];
cx q[3],q[4];
ry(0.6280727429783045) q[3];
ry(-1.5276284086051328) q[4];
cx q[3],q[4];
ry(2.238764268181912) q[4];
ry(-2.0241325898741076) q[5];
cx q[4],q[5];
ry(-2.2592815284233896) q[4];
ry(-1.8268609137606684) q[5];
cx q[4],q[5];
ry(1.933721425440094) q[5];
ry(-2.3349762013460866) q[6];
cx q[5],q[6];
ry(0.9391393699505617) q[5];
ry(-3.002004363094974) q[6];
cx q[5],q[6];
ry(-0.6933517872417703) q[6];
ry(1.0603791610420386) q[7];
cx q[6],q[7];
ry(-0.3474955412030084) q[6];
ry(1.5624403471148283) q[7];
cx q[6],q[7];
ry(2.4919470401915103) q[7];
ry(-2.5327892324613224) q[8];
cx q[7],q[8];
ry(0.6303119628104631) q[7];
ry(0.5757014626349549) q[8];
cx q[7],q[8];
ry(-2.32127287295979) q[8];
ry(2.757361583325467) q[9];
cx q[8],q[9];
ry(2.6037272374430884) q[8];
ry(2.6533503209999676) q[9];
cx q[8],q[9];
ry(2.3164062322369725) q[9];
ry(-1.5406053502786685) q[10];
cx q[9],q[10];
ry(-0.4933878349868826) q[9];
ry(-1.5611864146335304) q[10];
cx q[9],q[10];
ry(0.2571973814539129) q[10];
ry(3.0499170104250584) q[11];
cx q[10],q[11];
ry(1.9890482284332593) q[10];
ry(1.5507362968248442) q[11];
cx q[10],q[11];
ry(1.5331040849795992) q[11];
ry(-1.6728517466882886) q[12];
cx q[11],q[12];
ry(-2.8684988251114363) q[11];
ry(-0.4136194451035946) q[12];
cx q[11],q[12];
ry(-2.8877571233307897) q[12];
ry(-0.600497623985575) q[13];
cx q[12],q[13];
ry(-1.079482606099928) q[12];
ry(-1.2995641094956651) q[13];
cx q[12],q[13];
ry(-2.5038347904575255) q[13];
ry(2.889391863843649) q[14];
cx q[13],q[14];
ry(-1.5525124256214466) q[13];
ry(-1.7825144238880508) q[14];
cx q[13],q[14];
ry(2.563336100776527) q[14];
ry(1.4674367874573855) q[15];
cx q[14],q[15];
ry(-1.6459589413143245) q[14];
ry(0.6648920374816596) q[15];
cx q[14],q[15];
ry(1.0195195126556225) q[15];
ry(-2.4002352930823667) q[16];
cx q[15],q[16];
ry(-1.9437191372568243) q[15];
ry(2.297753570163234) q[16];
cx q[15],q[16];
ry(-3.1219813654349577) q[16];
ry(-0.5660104322460491) q[17];
cx q[16],q[17];
ry(1.9772347944491302) q[16];
ry(1.4226097461905316) q[17];
cx q[16],q[17];
ry(0.21245207576198818) q[17];
ry(-3.0496672930023476) q[18];
cx q[17],q[18];
ry(1.7047842334766221) q[17];
ry(-1.4433156281193196) q[18];
cx q[17],q[18];
ry(0.17169950666817968) q[18];
ry(1.0135081068993435) q[19];
cx q[18],q[19];
ry(3.088277855977741) q[18];
ry(-2.5762458047032677) q[19];
cx q[18],q[19];
ry(0.8982958391922065) q[0];
ry(0.07427162774910112) q[1];
cx q[0],q[1];
ry(2.997633809927061) q[0];
ry(1.061639131764804) q[1];
cx q[0],q[1];
ry(-1.4892888438774632) q[1];
ry(0.04888327113148324) q[2];
cx q[1],q[2];
ry(-1.4428711222769424) q[1];
ry(-1.3160226991769894) q[2];
cx q[1],q[2];
ry(2.796515737158841) q[2];
ry(-0.4607269328570691) q[3];
cx q[2],q[3];
ry(2.8660810559605934) q[2];
ry(1.1617222446969515) q[3];
cx q[2],q[3];
ry(-2.24433924160681) q[3];
ry(-2.0523122332341983) q[4];
cx q[3],q[4];
ry(-1.8377810352509139) q[3];
ry(-0.15617529837401084) q[4];
cx q[3],q[4];
ry(1.4383644299267662) q[4];
ry(3.089870249614765) q[5];
cx q[4],q[5];
ry(0.801376371212989) q[4];
ry(-1.8319244641468542) q[5];
cx q[4],q[5];
ry(-1.4290300521301953) q[5];
ry(2.0325353278012184) q[6];
cx q[5],q[6];
ry(-0.10469463302555089) q[5];
ry(-1.5337524574451447) q[6];
cx q[5],q[6];
ry(-1.996312800237616) q[6];
ry(-0.07375915898604557) q[7];
cx q[6],q[7];
ry(1.5558847036589736) q[6];
ry(0.4329348451795015) q[7];
cx q[6],q[7];
ry(1.6123004372250345) q[7];
ry(-1.0609525483934732) q[8];
cx q[7],q[8];
ry(1.5033803283041367) q[7];
ry(-1.3052174619602308) q[8];
cx q[7],q[8];
ry(3.0754954606348193) q[8];
ry(-0.3081097288938688) q[9];
cx q[8],q[9];
ry(-0.23273511538571243) q[8];
ry(-2.749080877236287) q[9];
cx q[8],q[9];
ry(3.0072522072986554) q[9];
ry(-0.0377000397985301) q[10];
cx q[9],q[10];
ry(-1.9683978451369262) q[9];
ry(0.41691581894990987) q[10];
cx q[9],q[10];
ry(0.013975103789519494) q[10];
ry(3.0464330655817067) q[11];
cx q[10],q[11];
ry(1.5577995204816568) q[10];
ry(1.535085253172189) q[11];
cx q[10],q[11];
ry(0.15557633747649383) q[11];
ry(-2.5437123180974868) q[12];
cx q[11],q[12];
ry(-1.5722781359489542) q[11];
ry(1.58382332339313) q[12];
cx q[11],q[12];
ry(1.5494930322857128) q[12];
ry(-0.06918206442957182) q[13];
cx q[12],q[13];
ry(-1.5931535772223455) q[12];
ry(2.7792059435601097) q[13];
cx q[12],q[13];
ry(-1.5839477911385043) q[13];
ry(-3.0358617808918016) q[14];
cx q[13],q[14];
ry(-1.5732282826778734) q[13];
ry(1.5598891646596718) q[14];
cx q[13],q[14];
ry(-1.543724793513963) q[14];
ry(0.00568139324938688) q[15];
cx q[14],q[15];
ry(-1.615682522446029) q[14];
ry(-2.7002534000084557) q[15];
cx q[14],q[15];
ry(1.563355265132382) q[15];
ry(0.6210879607121536) q[16];
cx q[15],q[16];
ry(-1.5718291257552433) q[15];
ry(1.5590681006278408) q[16];
cx q[15],q[16];
ry(-1.479980712154477) q[16];
ry(3.0669966183482753) q[17];
cx q[16],q[17];
ry(1.6581177598714198) q[16];
ry(-2.830500779936306) q[17];
cx q[16],q[17];
ry(1.5546955633605524) q[17];
ry(3.094156081575445) q[18];
cx q[17],q[18];
ry(1.5586046941128564) q[17];
ry(1.5677143082892118) q[18];
cx q[17],q[18];
ry(-1.7900018612911908) q[18];
ry(-0.5174104064531742) q[19];
cx q[18],q[19];
ry(1.41549298111281) q[18];
ry(0.5704672805711972) q[19];
cx q[18],q[19];
ry(2.7636498064392816) q[0];
ry(2.7585457985528388) q[1];
cx q[0],q[1];
ry(0.865465896586281) q[0];
ry(2.443043376437224) q[1];
cx q[0],q[1];
ry(1.7402783059161715) q[1];
ry(-0.09029389954553059) q[2];
cx q[1],q[2];
ry(-1.5524863688034145) q[1];
ry(1.469994705320245) q[2];
cx q[1],q[2];
ry(1.0432541253840855) q[2];
ry(-0.2206094414773334) q[3];
cx q[2],q[3];
ry(-0.019620126782446706) q[2];
ry(-3.1024240371100356) q[3];
cx q[2],q[3];
ry(-1.173570204413127) q[3];
ry(-1.3550485907137109) q[4];
cx q[3],q[4];
ry(0.1471006220195905) q[3];
ry(-1.4602719756626736) q[4];
cx q[3],q[4];
ry(1.978893605386439) q[4];
ry(-2.61502309120331) q[5];
cx q[4],q[5];
ry(1.0427777154704212) q[4];
ry(-3.1232014811704527) q[5];
cx q[4],q[5];
ry(-0.9331062040052869) q[5];
ry(0.020107789242209986) q[6];
cx q[5],q[6];
ry(-1.555109074100647) q[5];
ry(-1.6401513709102327) q[6];
cx q[5],q[6];
ry(0.6799876407181848) q[6];
ry(0.03577679509265064) q[7];
cx q[6],q[7];
ry(-2.9467418944477686) q[6];
ry(3.1222451926716306) q[7];
cx q[6],q[7];
ry(-3.0728110444626013) q[7];
ry(-0.07109229637645242) q[8];
cx q[7],q[8];
ry(-1.6183271692603138) q[7];
ry(1.3077262515192176) q[8];
cx q[7],q[8];
ry(-2.4498234356450963) q[8];
ry(2.885499669352737) q[9];
cx q[8],q[9];
ry(-3.0919608021641256) q[8];
ry(-3.0505316533836244) q[9];
cx q[8],q[9];
ry(-1.284656730687381) q[9];
ry(3.076009377631532) q[10];
cx q[9],q[10];
ry(1.5812049290421575) q[9];
ry(-0.014305189826688823) q[10];
cx q[9],q[10];
ry(2.9665759856688325) q[10];
ry(0.00024300143492883564) q[11];
cx q[10],q[11];
ry(1.6595483453024138) q[10];
ry(3.1306484765119804) q[11];
cx q[10],q[11];
ry(3.1149410641792783) q[11];
ry(-0.002521485041828064) q[12];
cx q[11],q[12];
ry(1.603558363687336) q[11];
ry(1.5649934018861127) q[12];
cx q[11],q[12];
ry(1.6665152294648746) q[12];
ry(-0.005323434447730868) q[13];
cx q[12],q[13];
ry(-1.5230411347115291) q[12];
ry(3.116081735877996) q[13];
cx q[12],q[13];
ry(3.1295516339982865) q[13];
ry(3.1403156768556006) q[14];
cx q[13],q[14];
ry(-1.568198036756038) q[13];
ry(1.568481555055655) q[14];
cx q[13],q[14];
ry(-1.3706662371620941) q[14];
ry(-0.0012046919783776058) q[15];
cx q[14],q[15];
ry(1.5436146171574117) q[14];
ry(3.136011414279411) q[15];
cx q[14],q[15];
ry(-3.1156508009095982) q[15];
ry(-3.1410929005972967) q[16];
cx q[15],q[16];
ry(-1.6219792407555955) q[15];
ry(1.5705603571498994) q[16];
cx q[15],q[16];
ry(-1.582097832152277) q[16];
ry(-0.014418583313158706) q[17];
cx q[16],q[17];
ry(1.5579465132414223) q[16];
ry(-0.01866512276706178) q[17];
cx q[16],q[17];
ry(2.990433404082971) q[17];
ry(-3.1246749585989937) q[18];
cx q[17],q[18];
ry(-1.6854564249411066) q[17];
ry(1.5702368645516462) q[18];
cx q[17],q[18];
ry(-2.2329281890247312) q[18];
ry(1.525204781560839) q[19];
cx q[18],q[19];
ry(-0.1635865080080494) q[18];
ry(2.9639914273741135) q[19];
cx q[18],q[19];
ry(-0.18588125025195446) q[0];
ry(1.2818675591488837) q[1];
ry(2.175130619343518) q[2];
ry(-0.2826466382064793) q[3];
ry(-1.2668836356333002) q[4];
ry(-0.22519629952370046) q[5];
ry(1.9232592106144504) q[6];
ry(-0.31612408879433024) q[7];
ry(1.2516530715630712) q[8];
ry(0.7127205886845828) q[9];
ry(-0.2579573911481683) q[10];
ry(2.852038337394353) q[11];
ry(-3.1324633547797265) q[12];
ry(2.850432944691223) q[13];
ry(0.0885061257787643) q[14];
ry(-0.291808689578418) q[15];
ry(3.04633171500803) q[16];
ry(2.8512360959132033) q[17];
ry(0.8097409838229171) q[18];
ry(2.8864455388151424) q[19];