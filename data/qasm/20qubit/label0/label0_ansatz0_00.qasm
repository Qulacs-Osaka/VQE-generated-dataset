OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
cx q[0],q[1];
rz(-0.09972717002890923) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.01793133907120685) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.07842627046247976) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.016642556590675987) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.02192235075434756) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.0261150640918297) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.002106865684564889) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.03892735208771068) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.04467698933181436) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.03394393189507802) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.05609237625040627) q[11];
cx q[10],q[11];
cx q[11],q[12];
rz(-0.08848607464382682) q[12];
cx q[11],q[12];
cx q[12],q[13];
rz(-0.0648244748181085) q[13];
cx q[12],q[13];
cx q[13],q[14];
rz(-0.08427262295845638) q[14];
cx q[13],q[14];
cx q[14],q[15];
rz(-0.07636887927919399) q[15];
cx q[14],q[15];
cx q[15],q[16];
rz(-0.04399024355939689) q[16];
cx q[15],q[16];
cx q[16],q[17];
rz(-0.05772779067995482) q[17];
cx q[16],q[17];
cx q[17],q[18];
rz(-0.09991907105026969) q[18];
cx q[17],q[18];
cx q[18],q[19];
rz(-0.025715411856923706) q[19];
cx q[18],q[19];
h q[0];
rz(2.987142438324367) q[0];
h q[0];
h q[1];
rz(1.9244857060437035) q[1];
h q[1];
h q[2];
rz(1.5676908026053302) q[2];
h q[2];
h q[3];
rz(1.576805913732141) q[3];
h q[3];
h q[4];
rz(1.5581454666182895) q[4];
h q[4];
h q[5];
rz(-1.5697114773894234) q[5];
h q[5];
h q[6];
rz(-1.5787473750798635) q[6];
h q[6];
h q[7];
rz(1.5312574293657244) q[7];
h q[7];
h q[8];
rz(1.5798962263762935) q[8];
h q[8];
h q[9];
rz(1.5747536455746762) q[9];
h q[9];
h q[10];
rz(1.575758717824731) q[10];
h q[10];
h q[11];
rz(1.5998485742393793) q[11];
h q[11];
h q[12];
rz(1.5976761028443451) q[12];
h q[12];
h q[13];
rz(-1.561867906054669) q[13];
h q[13];
h q[14];
rz(1.571525667393969) q[14];
h q[14];
h q[15];
rz(1.5650400891531018) q[15];
h q[15];
h q[16];
rz(-1.5793498771188272) q[16];
h q[16];
h q[17];
rz(-1.5880550435089926) q[17];
h q[17];
h q[18];
rz(-1.984979133204642) q[18];
h q[18];
h q[19];
rz(0.32327879874755805) q[19];
h q[19];
rz(1.6673292422159787) q[0];
rz(-0.09963235469747134) q[1];
rz(-1.7322733587919406) q[2];
rz(-1.5710367439873965) q[3];
rz(-1.5759811947787774) q[4];
rz(1.569507924537135) q[5];
rz(1.5725671629553404) q[6];
rz(-1.5737207101507198) q[7];
rz(1.5742301049767253) q[8];
rz(-1.566651365744961) q[9];
rz(-1.5676892952709365) q[10];
rz(-1.5418932059959716) q[11];
rz(-1.5416543861712941) q[12];
rz(1.5695199893037945) q[13];
rz(1.5708720636042395) q[14];
rz(-1.5764208849865644) q[15];
rz(1.5713774674100518) q[16];
rz(1.296636225375942) q[17];
rz(-0.9213263694552036) q[18];
rz(-1.2063701743724058) q[19];
cx q[0],q[1];
rz(1.6543168747441235) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.41362276135303544) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.17953151823026137) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(2.455812086014474) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(2.87170502630338) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(0.3837688214789904) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(0.7257054270140687) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(-1.5796297911282233) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(-2.02911038971725) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(0.9491457328702025) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(-2.2626124454391943) q[11];
cx q[10],q[11];
cx q[11],q[12];
rz(0.34571527535848307) q[12];
cx q[11],q[12];
cx q[12],q[13];
rz(-2.1925953414535937) q[13];
cx q[12],q[13];
cx q[13],q[14];
rz(2.3294400384037655) q[14];
cx q[13],q[14];
cx q[14],q[15];
rz(1.023919078661133) q[15];
cx q[14],q[15];
cx q[15],q[16];
rz(-0.8791118872388141) q[16];
cx q[15],q[16];
cx q[16],q[17];
rz(2.439365272259988) q[17];
cx q[16],q[17];
cx q[17],q[18];
rz(-0.6260679935596716) q[18];
cx q[17],q[18];
cx q[18],q[19];
rz(-2.4673071959645174) q[19];
cx q[18],q[19];
h q[0];
rz(0.785246602420079) q[0];
h q[0];
h q[1];
rz(-1.961462579086197) q[1];
h q[1];
h q[2];
rz(-0.5062869978510551) q[2];
h q[2];
h q[3];
rz(0.6925103015869939) q[3];
h q[3];
h q[4];
rz(0.47498754474168653) q[4];
h q[4];
h q[5];
rz(-2.804063151164282) q[5];
h q[5];
h q[6];
rz(-2.8127657773632353) q[6];
h q[6];
h q[7];
rz(-0.12556830966830704) q[7];
h q[7];
h q[8];
rz(-0.0631332512386863) q[8];
h q[8];
h q[9];
rz(0.07989194760992466) q[9];
h q[9];
h q[10];
rz(0.12662316427442066) q[10];
h q[10];
h q[11];
rz(-0.3015757832709576) q[11];
h q[11];
h q[12];
rz(-0.2733265096688395) q[12];
h q[12];
h q[13];
rz(-0.13135144634224655) q[13];
h q[13];
h q[14];
rz(-0.0980362218668239) q[14];
h q[14];
h q[15];
rz(-3.0471073285889374) q[15];
h q[15];
h q[16];
rz(-0.16894568625159778) q[16];
h q[16];
h q[17];
rz(-0.23191977929491508) q[17];
h q[17];
h q[18];
rz(2.6179081093735186) q[18];
h q[18];
h q[19];
rz(3.046064207643789) q[19];
h q[19];
rz(1.834267421907472) q[0];
rz(0.010341780708628756) q[1];
rz(-0.11234527849173626) q[2];
rz(-0.00019490501994772077) q[3];
rz(-0.0018802229920510225) q[4];
rz(-0.004808979705352523) q[5];
rz(-0.0022500570435329457) q[6];
rz(-0.00019682687752570172) q[7];
rz(0.0030486185697194) q[8];
rz(-0.002308092967064954) q[9];
rz(0.00018307596427534828) q[10];
rz(0.0011479048804103442) q[11];
rz(0.0016120661669934769) q[12];
rz(0.0015238855806214559) q[13];
rz(-0.00048718491422130846) q[14];
rz(0.0014315184007585762) q[15];
rz(0.0007156728092292958) q[16];
rz(0.0009120568261631231) q[17];
rz(0.0031202766318595724) q[18];
rz(-2.2479845487016963) q[19];
cx q[0],q[1];
rz(0.5545319458087047) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.437632237279846) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.12418987163653664) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(-2.285849606436533) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(2.642664737277383) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.13121200409164868) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(0.5772787019366625) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(1.7838785772064896) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(2.2724670259077913) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(-1.1619696194449147) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(2.4484185354445245) q[11];
cx q[10],q[11];
cx q[11],q[12];
rz(-0.10238337126537941) q[12];
cx q[11],q[12];
cx q[12],q[13];
rz(2.396443818587247) q[13];
cx q[12],q[13];
cx q[13],q[14];
rz(1.0233132326692054) q[14];
cx q[13],q[14];
cx q[14],q[15];
rz(0.7789281695553808) q[15];
cx q[14],q[15];
cx q[15],q[16];
rz(-0.6730765075941026) q[16];
cx q[15],q[16];
cx q[16],q[17];
rz(0.541799839167767) q[17];
cx q[16],q[17];
cx q[17],q[18];
rz(-0.31705281899641413) q[18];
cx q[17],q[18];
cx q[18],q[19];
rz(-1.3918143969723193) q[19];
cx q[18],q[19];
h q[0];
rz(1.7686388577892773) q[0];
h q[0];
h q[1];
rz(-0.5437635599138887) q[1];
h q[1];
h q[2];
rz(0.016699825303579274) q[2];
h q[2];
h q[3];
rz(-1.0078622697511541) q[3];
h q[3];
h q[4];
rz(1.8928867591068583) q[4];
h q[4];
h q[5];
rz(2.8532662825478914) q[5];
h q[5];
h q[6];
rz(-1.2022684221590572) q[6];
h q[6];
h q[7];
rz(-1.4444178073657195) q[7];
h q[7];
h q[8];
rz(-2.433099882501678) q[8];
h q[8];
h q[9];
rz(-0.7476161293846623) q[9];
h q[9];
h q[10];
rz(-1.2728089636014968) q[10];
h q[10];
h q[11];
rz(2.0354706959705866) q[11];
h q[11];
h q[12];
rz(-1.023478021016496) q[12];
h q[12];
h q[13];
rz(-1.9315485972631012) q[13];
h q[13];
h q[14];
rz(-0.8056694644553959) q[14];
h q[14];
h q[15];
rz(-0.7978722633436004) q[15];
h q[15];
h q[16];
rz(-1.7101959558895186) q[16];
h q[16];
h q[17];
rz(-1.9834600044913617) q[17];
h q[17];
h q[18];
rz(-0.5634139175007625) q[18];
h q[18];
h q[19];
rz(-1.4663045387312978) q[19];
h q[19];
rz(0.8893562603660817) q[0];
rz(-0.026076423542047155) q[1];
rz(0.11481092872456827) q[2];
rz(-0.012608780603283815) q[3];
rz(-0.012821153181996808) q[4];
rz(-0.0052416624368542665) q[5];
rz(0.01816775780859152) q[6];
rz(0.04323313369601422) q[7];
rz(-3.1247738721061284) q[8];
rz(-0.006902923749332316) q[9];
rz(-0.007449708398407375) q[10];
rz(0.03586782974581167) q[11];
rz(-0.038167650421973) q[12];
rz(3.133781251346999) q[13];
rz(-0.0009339265460022047) q[14];
rz(-0.008443643105393894) q[15];
rz(-3.131707363143178) q[16];
rz(-3.1192002410075514) q[17];
rz(-0.005694952725680138) q[18];
rz(-1.3656376038426343) q[19];