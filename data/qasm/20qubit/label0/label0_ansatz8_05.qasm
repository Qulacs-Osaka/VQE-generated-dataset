OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-1.8870574244707552) q[0];
ry(-0.16008608302163987) q[1];
cx q[0],q[1];
ry(1.490337048956481) q[0];
ry(-1.5418061483370984) q[1];
cx q[0],q[1];
ry(2.55436766498196) q[2];
ry(1.7364701774621683) q[3];
cx q[2],q[3];
ry(-3.1297853462616168) q[2];
ry(3.1381341188920957) q[3];
cx q[2],q[3];
ry(2.2589766620811194) q[4];
ry(-0.7059042479741682) q[5];
cx q[4],q[5];
ry(1.963171078630172) q[4];
ry(-1.6917913748162787) q[5];
cx q[4],q[5];
ry(2.8324791358848307) q[6];
ry(-0.16694606911020404) q[7];
cx q[6],q[7];
ry(0.4006457483742944) q[6];
ry(-2.2874819698216253) q[7];
cx q[6],q[7];
ry(0.1985944146322769) q[8];
ry(-0.28970033032217796) q[9];
cx q[8],q[9];
ry(2.972208223995068) q[8];
ry(-0.5788587592017934) q[9];
cx q[8],q[9];
ry(-2.4501967615858273) q[10];
ry(-0.580872506957519) q[11];
cx q[10],q[11];
ry(0.9126021485709668) q[10];
ry(-1.4868615073092049) q[11];
cx q[10],q[11];
ry(-0.17413372331264743) q[12];
ry(2.092863201001324) q[13];
cx q[12],q[13];
ry(-0.10628744708782503) q[12];
ry(2.942955444801914) q[13];
cx q[12],q[13];
ry(-2.748919136829792) q[14];
ry(1.5225525816133654) q[15];
cx q[14],q[15];
ry(-0.3216025148533823) q[14];
ry(-2.9327926905452952) q[15];
cx q[14],q[15];
ry(-1.9994225912804762) q[16];
ry(1.2087744895274417) q[17];
cx q[16],q[17];
ry(-0.33562393928582157) q[16];
ry(0.1588536451294438) q[17];
cx q[16],q[17];
ry(0.87413100733219) q[18];
ry(-2.189754736621763) q[19];
cx q[18],q[19];
ry(-2.525716075784699) q[18];
ry(-0.8033030597587247) q[19];
cx q[18],q[19];
ry(-2.0933997054525593) q[0];
ry(0.8744377829329738) q[2];
cx q[0],q[2];
ry(-2.1849910975159075) q[0];
ry(2.766094853144687) q[2];
cx q[0],q[2];
ry(0.40699002119309036) q[2];
ry(-2.3971868488638086) q[4];
cx q[2],q[4];
ry(-2.6583749599266784) q[2];
ry(-2.6550883635161173) q[4];
cx q[2],q[4];
ry(0.9576695037904673) q[4];
ry(1.1284282271305643) q[6];
cx q[4],q[6];
ry(-1.623116085783363) q[4];
ry(0.11029770128665815) q[6];
cx q[4],q[6];
ry(2.9348001390094063) q[6];
ry(-1.9237223075059422) q[8];
cx q[6],q[8];
ry(0.0775015834880464) q[6];
ry(-3.1357009183522457) q[8];
cx q[6],q[8];
ry(-1.5196458909521215) q[8];
ry(-1.1482452301532506) q[10];
cx q[8],q[10];
ry(0.0017967519542532173) q[8];
ry(-0.1467850691874289) q[10];
cx q[8],q[10];
ry(2.5126148026585335) q[10];
ry(-1.811543244584775) q[12];
cx q[10],q[12];
ry(2.4077894819691017) q[10];
ry(0.12040538148818455) q[12];
cx q[10],q[12];
ry(-1.4139798263321812) q[12];
ry(-1.2982269612188064) q[14];
cx q[12],q[14];
ry(1.2047992360249955) q[12];
ry(-2.943213464238457) q[14];
cx q[12],q[14];
ry(2.094993184419723) q[14];
ry(-2.3736729037221567) q[16];
cx q[14],q[16];
ry(0.3885832059441634) q[14];
ry(-0.01222875732506955) q[16];
cx q[14],q[16];
ry(-0.2919478179614634) q[16];
ry(-2.219920530281712) q[18];
cx q[16],q[18];
ry(0.18068644296309117) q[16];
ry(-3.1120605170118694) q[18];
cx q[16],q[18];
ry(-0.6014775500638132) q[1];
ry(-2.3642465494111926) q[3];
cx q[1],q[3];
ry(-2.031491657386384) q[1];
ry(-2.744878212343436) q[3];
cx q[1],q[3];
ry(-1.3914981641609834) q[3];
ry(-0.979240910802595) q[5];
cx q[3],q[5];
ry(-1.488102946534756) q[3];
ry(-3.1415532113496623) q[5];
cx q[3],q[5];
ry(1.7539784963234402) q[5];
ry(1.3112690943181669) q[7];
cx q[5],q[7];
ry(-3.0951322297723896) q[5];
ry(-0.02101012612881165) q[7];
cx q[5],q[7];
ry(-0.996950944682542) q[7];
ry(1.2733805744618882) q[9];
cx q[7],q[9];
ry(-0.008776174320257901) q[7];
ry(-3.138792846000797) q[9];
cx q[7],q[9];
ry(0.41680681404677217) q[9];
ry(-2.2622894122725006) q[11];
cx q[9],q[11];
ry(1.1425573161743783) q[9];
ry(3.0923106812245202) q[11];
cx q[9],q[11];
ry(1.5970839112905626) q[11];
ry(-1.5941659912369932) q[13];
cx q[11],q[13];
ry(-0.06149032005399491) q[11];
ry(-1.3245787634426054) q[13];
cx q[11],q[13];
ry(-2.8924607271231997) q[13];
ry(-1.6275345106935228) q[15];
cx q[13],q[15];
ry(0.20991062606069386) q[13];
ry(2.9284711866570787) q[15];
cx q[13],q[15];
ry(-2.738869938560473) q[15];
ry(1.6810013271633517) q[17];
cx q[15],q[17];
ry(3.1352321012767597) q[15];
ry(-0.03805755631732666) q[17];
cx q[15],q[17];
ry(1.2969746423103816) q[17];
ry(-0.7671012699885358) q[19];
cx q[17],q[19];
ry(-1.4684601725393485) q[17];
ry(1.740760969207149) q[19];
cx q[17],q[19];
ry(-2.7643391667795463) q[0];
ry(-1.3629333279589462) q[1];
cx q[0],q[1];
ry(2.312920216069548) q[0];
ry(-1.717969086778369) q[1];
cx q[0],q[1];
ry(2.741756188024287) q[2];
ry(2.2408093994190095) q[3];
cx q[2],q[3];
ry(0.1256519003675233) q[2];
ry(-0.07829199230177077) q[3];
cx q[2],q[3];
ry(0.6838170294575203) q[4];
ry(2.4467268821512653) q[5];
cx q[4],q[5];
ry(-2.9386693261897006) q[4];
ry(-2.8450433647299014) q[5];
cx q[4],q[5];
ry(0.9151828106795944) q[6];
ry(1.7114589985560595) q[7];
cx q[6],q[7];
ry(0.5229271047994573) q[6];
ry(-2.378674544534953) q[7];
cx q[6],q[7];
ry(-1.2890791116430838) q[8];
ry(-2.0281485684231733) q[9];
cx q[8],q[9];
ry(-2.6175222674132845) q[8];
ry(-0.5464548658674051) q[9];
cx q[8],q[9];
ry(-3.1212743042801425) q[10];
ry(-1.9301804152902504) q[11];
cx q[10],q[11];
ry(-1.2189000896793027) q[10];
ry(0.28692108147019013) q[11];
cx q[10],q[11];
ry(1.6739840329406226) q[12];
ry(-0.9563711288035222) q[13];
cx q[12],q[13];
ry(0.3322780975749899) q[12];
ry(0.7377217468004682) q[13];
cx q[12],q[13];
ry(-2.693362425413132) q[14];
ry(2.402562678539679) q[15];
cx q[14],q[15];
ry(-1.7082674893358913) q[14];
ry(-1.8484224650886256) q[15];
cx q[14],q[15];
ry(-2.324849637335896) q[16];
ry(3.109885207675098) q[17];
cx q[16],q[17];
ry(-0.7508174175755405) q[16];
ry(-0.722197497677115) q[17];
cx q[16],q[17];
ry(1.7295055120927485) q[18];
ry(0.06127602896950534) q[19];
cx q[18],q[19];
ry(-0.098108188658661) q[18];
ry(1.6414976485622907) q[19];
cx q[18],q[19];
ry(-0.17148603455249312) q[0];
ry(0.41419858494401196) q[2];
cx q[0],q[2];
ry(0.40432730092335856) q[0];
ry(-3.1345181443088124) q[2];
cx q[0],q[2];
ry(-1.4029736986976014) q[2];
ry(1.6596447978909952) q[4];
cx q[2],q[4];
ry(3.140994813946369) q[2];
ry(3.141541292379895) q[4];
cx q[2],q[4];
ry(2.942414804311882) q[4];
ry(-1.8909574249324494) q[6];
cx q[4],q[6];
ry(3.056600570926061) q[4];
ry(0.08925705513739125) q[6];
cx q[4],q[6];
ry(2.3703558268454588) q[6];
ry(-2.3009956248015873) q[8];
cx q[6],q[8];
ry(0.017718414151589364) q[6];
ry(-0.04979012142238404) q[8];
cx q[6],q[8];
ry(1.6327199509867292) q[8];
ry(0.3092428964064295) q[10];
cx q[8],q[10];
ry(-2.406987225282473) q[8];
ry(-0.553482050491583) q[10];
cx q[8],q[10];
ry(0.42283925294854674) q[10];
ry(0.7432471068255171) q[12];
cx q[10],q[12];
ry(-3.1263229650123256) q[10];
ry(0.00605643396084421) q[12];
cx q[10],q[12];
ry(1.3757412764935575) q[12];
ry(-1.4934444865874388) q[14];
cx q[12],q[14];
ry(0.02696568961893044) q[12];
ry(2.9968605744230543) q[14];
cx q[12],q[14];
ry(2.2596930042139745) q[14];
ry(-0.5180928712087532) q[16];
cx q[14],q[16];
ry(2.4379190173621295) q[14];
ry(2.885628157901775) q[16];
cx q[14],q[16];
ry(2.9734734255296877) q[16];
ry(1.4727959934942891) q[18];
cx q[16],q[18];
ry(2.465813061984325) q[16];
ry(3.141341987402481) q[18];
cx q[16],q[18];
ry(-1.9079692081214956) q[1];
ry(-0.7355895606914821) q[3];
cx q[1],q[3];
ry(1.2873121746399452) q[1];
ry(-2.8868165240046744) q[3];
cx q[1],q[3];
ry(1.5406819694043272) q[3];
ry(-2.0466846688450944) q[5];
cx q[3],q[5];
ry(0.00246262856791013) q[3];
ry(0.000692475686500771) q[5];
cx q[3],q[5];
ry(-1.5607746745582831) q[5];
ry(-2.1313129248293006) q[7];
cx q[5],q[7];
ry(1.1501896710504909) q[5];
ry(3.087837572503046) q[7];
cx q[5],q[7];
ry(2.119089394796862) q[7];
ry(1.4727855959732148) q[9];
cx q[7],q[9];
ry(0.9165033996323801) q[7];
ry(-3.1335588247597825) q[9];
cx q[7],q[9];
ry(-2.424467313585007) q[9];
ry(1.8381942388656218) q[11];
cx q[9],q[11];
ry(-0.030114530559961432) q[9];
ry(0.0392282873626435) q[11];
cx q[9],q[11];
ry(-0.961880869374257) q[11];
ry(-0.24884874103280225) q[13];
cx q[11],q[13];
ry(3.039153954339474) q[11];
ry(-3.032831124584451) q[13];
cx q[11],q[13];
ry(-0.21803907372329248) q[13];
ry(-0.9477654436085583) q[15];
cx q[13],q[15];
ry(-0.07962897110526157) q[13];
ry(3.001372205416828) q[15];
cx q[13],q[15];
ry(-0.35290043406921906) q[15];
ry(-2.3156302292046957) q[17];
cx q[15],q[17];
ry(0.6742723984581707) q[15];
ry(3.0281256248196913) q[17];
cx q[15],q[17];
ry(-1.9142961027937335) q[17];
ry(2.2928712471321226) q[19];
cx q[17],q[19];
ry(2.1083232101019944) q[17];
ry(0.7747122677879084) q[19];
cx q[17],q[19];
ry(-1.764865548332125) q[0];
ry(1.0733563518124105) q[1];
cx q[0],q[1];
ry(1.0560972222346994) q[0];
ry(0.811046409434014) q[1];
cx q[0],q[1];
ry(-0.902758341026556) q[2];
ry(-1.5963807336227127) q[3];
cx q[2],q[3];
ry(-2.9643551364799507) q[2];
ry(-2.7473915340446937) q[3];
cx q[2],q[3];
ry(1.0275886330029635) q[4];
ry(-2.3199401946444) q[5];
cx q[4],q[5];
ry(-2.2746988341847265) q[4];
ry(-1.3598178827669125) q[5];
cx q[4],q[5];
ry(2.728511724239348) q[6];
ry(-2.04738147366147) q[7];
cx q[6],q[7];
ry(2.984161450485388) q[6];
ry(-1.4971658841914037) q[7];
cx q[6],q[7];
ry(-1.7561910969660381) q[8];
ry(-1.205329739077975) q[9];
cx q[8],q[9];
ry(1.6497661904991587) q[8];
ry(0.7517159106522493) q[9];
cx q[8],q[9];
ry(1.6554073345915112) q[10];
ry(2.355017050750957) q[11];
cx q[10],q[11];
ry(1.4858095690348438) q[10];
ry(-1.4979659290488936) q[11];
cx q[10],q[11];
ry(-0.5658257468575782) q[12];
ry(1.0557552202496603) q[13];
cx q[12],q[13];
ry(-1.9419890540829474) q[12];
ry(-2.2396793816144775) q[13];
cx q[12],q[13];
ry(1.7164842596028342) q[14];
ry(2.399555708032796) q[15];
cx q[14],q[15];
ry(-2.2949140500944267) q[14];
ry(0.7909864811627713) q[15];
cx q[14],q[15];
ry(-0.8554721601533597) q[16];
ry(2.156288695649998) q[17];
cx q[16],q[17];
ry(-0.034849459602192365) q[16];
ry(-3.101074012984847) q[17];
cx q[16],q[17];
ry(-2.350576819849818) q[18];
ry(0.9214373390050294) q[19];
cx q[18],q[19];
ry(-0.0011610141150650345) q[18];
ry(0.7887061857216304) q[19];
cx q[18],q[19];
ry(-3.1378870112945574) q[0];
ry(-1.3352797404271257) q[2];
cx q[0],q[2];
ry(0.9913492018482639) q[0];
ry(0.17900675247530273) q[2];
cx q[0],q[2];
ry(-2.1113103772321598) q[2];
ry(1.3362005508029675) q[4];
cx q[2],q[4];
ry(-3.1411902566724987) q[2];
ry(6.427413790783557e-05) q[4];
cx q[2],q[4];
ry(-2.1615148877171677) q[4];
ry(1.5933476254670476) q[6];
cx q[4],q[6];
ry(-2.7726894557403416) q[4];
ry(0.0762068968527037) q[6];
cx q[4],q[6];
ry(2.9989803327374314) q[6];
ry(1.8357952865724256) q[8];
cx q[6],q[8];
ry(0.7781745021749256) q[6];
ry(0.731414593892719) q[8];
cx q[6],q[8];
ry(1.448176465479844) q[8];
ry(-0.06906017835971229) q[10];
cx q[8],q[10];
ry(-2.6871117652104997) q[8];
ry(3.139273738803155) q[10];
cx q[8],q[10];
ry(3.1223116617079167) q[10];
ry(1.8431985137615463) q[12];
cx q[10],q[12];
ry(0.8802134630900298) q[10];
ry(-2.29494760884684) q[12];
cx q[10],q[12];
ry(-0.23324421852103996) q[12];
ry(-1.2817968037401302) q[14];
cx q[12],q[14];
ry(3.141174269191183) q[12];
ry(0.14157694983924252) q[14];
cx q[12],q[14];
ry(-0.7604797751958636) q[14];
ry(0.44470602172162454) q[16];
cx q[14],q[16];
ry(1.6056956282964652) q[14];
ry(0.6241101179095634) q[16];
cx q[14],q[16];
ry(-2.55659537004295) q[16];
ry(1.9362401776440823) q[18];
cx q[16],q[18];
ry(-4.981100648105429e-05) q[16];
ry(-0.00010780233036271134) q[18];
cx q[16],q[18];
ry(-2.9739775075262274) q[1];
ry(1.3454007788507232) q[3];
cx q[1],q[3];
ry(-2.554384270743171) q[1];
ry(-1.4184073796915022) q[3];
cx q[1],q[3];
ry(-0.7248899062852469) q[3];
ry(-0.5798299374142875) q[5];
cx q[3],q[5];
ry(0.00011013901880662382) q[3];
ry(2.9161139270428297e-05) q[5];
cx q[3],q[5];
ry(2.4968009894297927) q[5];
ry(2.781109475979716) q[7];
cx q[5],q[7];
ry(-0.45600697073195035) q[5];
ry(-3.1394719357013434) q[7];
cx q[5],q[7];
ry(0.5721901938522872) q[7];
ry(-2.7058860241007725) q[9];
cx q[7],q[9];
ry(-0.0793815843541461) q[7];
ry(3.05316660906701) q[9];
cx q[7],q[9];
ry(2.334604007126542) q[9];
ry(2.4966415258200665) q[11];
cx q[9],q[11];
ry(-0.8067917941361544) q[9];
ry(-2.4839755982133314) q[11];
cx q[9],q[11];
ry(2.9308208545091903) q[11];
ry(-1.772457524296592) q[13];
cx q[11],q[13];
ry(-3.1369844752151654) q[11];
ry(0.010064245382014114) q[13];
cx q[11],q[13];
ry(0.47307794028437955) q[13];
ry(1.831954601260277) q[15];
cx q[13],q[15];
ry(-3.0505287287147613) q[13];
ry(-2.5468023552142993) q[15];
cx q[13],q[15];
ry(2.31606567775978) q[15];
ry(-2.3719430184974133) q[17];
cx q[15],q[17];
ry(-0.6270731189358036) q[15];
ry(3.0138116784105304) q[17];
cx q[15],q[17];
ry(-1.0707881479956036) q[17];
ry(-2.517292415801864) q[19];
cx q[17],q[19];
ry(0.0016845984628748312) q[17];
ry(-0.004378024529133384) q[19];
cx q[17],q[19];
ry(3.1406582884386527) q[0];
ry(1.9545244485953164) q[1];
cx q[0],q[1];
ry(-1.1657714242663646) q[0];
ry(0.10107174080509851) q[1];
cx q[0],q[1];
ry(-1.2692125195508126) q[2];
ry(-0.6913237190398498) q[3];
cx q[2],q[3];
ry(3.0635764592218964) q[2];
ry(-0.10687027926979376) q[3];
cx q[2],q[3];
ry(-1.262736261923677) q[4];
ry(2.382067278790994) q[5];
cx q[4],q[5];
ry(0.5157851920784752) q[4];
ry(0.7733954993421783) q[5];
cx q[4],q[5];
ry(-2.877650851905961) q[6];
ry(-1.8877334551565896) q[7];
cx q[6],q[7];
ry(-2.6884304040414793) q[6];
ry(-0.3467398119896171) q[7];
cx q[6],q[7];
ry(-1.288659026849628) q[8];
ry(-2.3401838076025907) q[9];
cx q[8],q[9];
ry(2.733449137746871) q[8];
ry(-1.266907980815268) q[9];
cx q[8],q[9];
ry(-0.8704431865833753) q[10];
ry(-2.9093723600629136) q[11];
cx q[10],q[11];
ry(0.06383952852259075) q[10];
ry(3.032000459573199) q[11];
cx q[10],q[11];
ry(1.0602584920009874) q[12];
ry(-2.6100446016277217) q[13];
cx q[12],q[13];
ry(-0.0007916959496294014) q[12];
ry(0.0011149710144886527) q[13];
cx q[12],q[13];
ry(0.40329597210094636) q[14];
ry(-1.3762389284993686) q[15];
cx q[14],q[15];
ry(1.9073588310471568) q[14];
ry(-2.1201294927330023) q[15];
cx q[14],q[15];
ry(-2.0123690896450563) q[16];
ry(0.9625660723786966) q[17];
cx q[16],q[17];
ry(-2.5590466663076508) q[16];
ry(-3.026981790609071) q[17];
cx q[16],q[17];
ry(-0.45306703272454873) q[18];
ry(0.31413401793539997) q[19];
cx q[18],q[19];
ry(0.09783544247691085) q[18];
ry(-2.6014960667749785) q[19];
cx q[18],q[19];
ry(0.48983432875059196) q[0];
ry(0.6415510121505319) q[2];
cx q[0],q[2];
ry(2.771400619920903) q[0];
ry(1.9701642154874828) q[2];
cx q[0],q[2];
ry(-2.7368386221239147) q[2];
ry(2.926712472091354) q[4];
cx q[2],q[4];
ry(3.1413829572117837) q[2];
ry(3.010244003637428e-07) q[4];
cx q[2],q[4];
ry(2.0425458504937084) q[4];
ry(2.4604343622459823) q[6];
cx q[4],q[6];
ry(3.1400201159999557) q[4];
ry(-1.7478087377725506) q[6];
cx q[4],q[6];
ry(-1.165162758080435) q[6];
ry(1.929673435938958) q[8];
cx q[6],q[8];
ry(-2.0750065040341026) q[6];
ry(-0.41274843286666485) q[8];
cx q[6],q[8];
ry(-0.46445480018469976) q[8];
ry(-0.9917923871522958) q[10];
cx q[8],q[10];
ry(-3.1304547867591173) q[8];
ry(3.1387963078284815) q[10];
cx q[8],q[10];
ry(-2.5057345301419045) q[10];
ry(1.2265886805007264) q[12];
cx q[10],q[12];
ry(2.6992494554547197) q[10];
ry(1.3401748448369604) q[12];
cx q[10],q[12];
ry(-0.7209173681753391) q[12];
ry(0.7741911509913784) q[14];
cx q[12],q[14];
ry(3.1406355717812056) q[12];
ry(3.1415048970847077) q[14];
cx q[12],q[14];
ry(-0.2514026973443099) q[14];
ry(2.533274550785918) q[16];
cx q[14],q[16];
ry(1.4462256040982888) q[14];
ry(-2.741085549970699) q[16];
cx q[14],q[16];
ry(-2.671380834460766) q[16];
ry(-0.6404566930002655) q[18];
cx q[16],q[18];
ry(-2.232524831733264) q[16];
ry(-3.141308264450948) q[18];
cx q[16],q[18];
ry(1.7995843616807035) q[1];
ry(-1.647397460746312) q[3];
cx q[1],q[3];
ry(2.8340104498798) q[1];
ry(1.2623822317375255) q[3];
cx q[1],q[3];
ry(0.10602309862179293) q[3];
ry(-1.7767182626457305) q[5];
cx q[3],q[5];
ry(-3.141078306413229) q[3];
ry(-3.141400018696061) q[5];
cx q[3],q[5];
ry(-2.3642153086747695) q[5];
ry(-2.6750187800792933) q[7];
cx q[5],q[7];
ry(2.9839508495920715) q[5];
ry(2.584090640190257) q[7];
cx q[5],q[7];
ry(-2.8188916209372694) q[7];
ry(0.5164905588561675) q[9];
cx q[7],q[9];
ry(-0.06430020432482834) q[7];
ry(-0.05996299074062783) q[9];
cx q[7],q[9];
ry(-0.040264791712216995) q[9];
ry(-2.3220039039032874) q[11];
cx q[9],q[11];
ry(1.702279584672714) q[9];
ry(-0.0012106680819151934) q[11];
cx q[9],q[11];
ry(1.3168771482774146) q[11];
ry(-2.336095981292481) q[13];
cx q[11],q[13];
ry(0.2436244162013157) q[11];
ry(0.04117655169786742) q[13];
cx q[11],q[13];
ry(1.49346369898932) q[13];
ry(-1.1728923361087622) q[15];
cx q[13],q[15];
ry(0.03549721676423072) q[13];
ry(-0.314224089532663) q[15];
cx q[13],q[15];
ry(1.3768876695254093) q[15];
ry(-0.22328009144172611) q[17];
cx q[15],q[17];
ry(-2.885192084680495) q[15];
ry(-0.1376957711864666) q[17];
cx q[15],q[17];
ry(-1.1099629782209945) q[17];
ry(-1.0699616414437312) q[19];
cx q[17],q[19];
ry(1.9025541093949068) q[17];
ry(-0.012302254670658163) q[19];
cx q[17],q[19];
ry(-1.6962631163151294) q[0];
ry(1.2054396434372443) q[1];
cx q[0],q[1];
ry(-0.11296139396420735) q[0];
ry(2.1927686212484936) q[1];
cx q[0],q[1];
ry(2.8275433456112866) q[2];
ry(-0.24883512500176508) q[3];
cx q[2],q[3];
ry(-0.8267978460263796) q[2];
ry(0.9283971366670702) q[3];
cx q[2],q[3];
ry(-0.07714183267932612) q[4];
ry(-2.0293773926182053) q[5];
cx q[4],q[5];
ry(-3.0844800480590644) q[4];
ry(0.47287114851065803) q[5];
cx q[4],q[5];
ry(2.8378430657644222) q[6];
ry(1.3987732530500199) q[7];
cx q[6],q[7];
ry(2.9654656022280914) q[6];
ry(3.0080667624776476) q[7];
cx q[6],q[7];
ry(1.2104933953744175) q[8];
ry(1.8717152348007957) q[9];
cx q[8],q[9];
ry(-1.4667171108025743) q[8];
ry(-1.6743253043109751) q[9];
cx q[8],q[9];
ry(0.48330825612700323) q[10];
ry(0.5023521040001806) q[11];
cx q[10],q[11];
ry(-3.140110538173048) q[10];
ry(-2.39823278075042) q[11];
cx q[10],q[11];
ry(-2.4663011508341017) q[12];
ry(2.6327551808056486) q[13];
cx q[12],q[13];
ry(-3.1395782354147355) q[12];
ry(-3.138455654202174) q[13];
cx q[12],q[13];
ry(2.86415363363058) q[14];
ry(-2.2288644787589833) q[15];
cx q[14],q[15];
ry(1.0113605714216032) q[14];
ry(-2.3121721219729383) q[15];
cx q[14],q[15];
ry(3.0671312495846186) q[16];
ry(2.024600878422688) q[17];
cx q[16],q[17];
ry(-2.715328858246119) q[16];
ry(-3.1326831419648946) q[17];
cx q[16],q[17];
ry(3.1212769907741653) q[18];
ry(-2.616617551024247) q[19];
cx q[18],q[19];
ry(-1.5098536513626835) q[18];
ry(1.5725408211032068) q[19];
cx q[18],q[19];
ry(0.7862329531910941) q[0];
ry(0.9784249169281202) q[2];
cx q[0],q[2];
ry(-0.06275626844565452) q[0];
ry(0.012356560503478775) q[2];
cx q[0],q[2];
ry(-2.078461253750694) q[2];
ry(-2.4190418586399107) q[4];
cx q[2],q[4];
ry(-9.234857184968641e-05) q[2];
ry(-3.141514155082136) q[4];
cx q[2],q[4];
ry(-0.7336789851718652) q[4];
ry(-0.2781662490795105) q[6];
cx q[4],q[6];
ry(0.05324614900194901) q[4];
ry(-0.7458690238142226) q[6];
cx q[4],q[6];
ry(-2.67649680421204) q[6];
ry(1.493327343826251) q[8];
cx q[6],q[8];
ry(-2.4502073592517033) q[6];
ry(-1.613068852603962) q[8];
cx q[6],q[8];
ry(2.0303876053301297) q[8];
ry(-1.5572388309339589) q[10];
cx q[8],q[10];
ry(0.001433786105277335) q[8];
ry(-0.0005619049169551542) q[10];
cx q[8],q[10];
ry(1.2563596720124401) q[10];
ry(3.0551049764135314) q[12];
cx q[10],q[12];
ry(0.5965553115420902) q[10];
ry(-2.0391061942336135) q[12];
cx q[10],q[12];
ry(-0.6881368925816932) q[12];
ry(1.5550905695592405) q[14];
cx q[12],q[14];
ry(3.1406806679862687) q[12];
ry(3.1412878680408025) q[14];
cx q[12],q[14];
ry(3.035711317042228) q[14];
ry(2.238537136303117) q[16];
cx q[14],q[16];
ry(-1.462452085140405) q[14];
ry(3.127393181711262) q[16];
cx q[14],q[16];
ry(1.309789388920838) q[16];
ry(2.6318053327405346) q[18];
cx q[16],q[18];
ry(-0.001258950780988144) q[16];
ry(0.00730824422938122) q[18];
cx q[16],q[18];
ry(-2.340904708962409) q[1];
ry(0.9042936000176107) q[3];
cx q[1],q[3];
ry(2.6339000321726105) q[1];
ry(1.4160427585019943) q[3];
cx q[1],q[3];
ry(-0.2876192373490696) q[3];
ry(2.7086854616635536) q[5];
cx q[3],q[5];
ry(0.0006168380407079876) q[3];
ry(0.0003064831222999772) q[5];
cx q[3],q[5];
ry(-0.41044862575270624) q[5];
ry(2.619628022937692) q[7];
cx q[5],q[7];
ry(3.129384880891963) q[5];
ry(-3.026428687777055) q[7];
cx q[5],q[7];
ry(-3.002378493558222) q[7];
ry(1.9178561654553297) q[9];
cx q[7],q[9];
ry(-1.3902370178853705) q[7];
ry(1.6991257008607656) q[9];
cx q[7],q[9];
ry(2.857751650862306) q[9];
ry(-2.4439317530207867) q[11];
cx q[9],q[11];
ry(-3.1347557117981655) q[9];
ry(3.136452295317221) q[11];
cx q[9],q[11];
ry(0.8470463615052976) q[11];
ry(-2.104965632050561) q[13];
cx q[11],q[13];
ry(-2.9032398684573737) q[11];
ry(-0.001948837180865728) q[13];
cx q[11],q[13];
ry(0.42515940181124107) q[13];
ry(-2.857016840402127) q[15];
cx q[13],q[15];
ry(-1.430680865492497) q[13];
ry(3.096832634531293) q[15];
cx q[13],q[15];
ry(1.3152542020153213) q[15];
ry(-2.1752661968431317) q[17];
cx q[15],q[17];
ry(-2.483005025522211) q[15];
ry(-2.8596815593702964) q[17];
cx q[15],q[17];
ry(2.2282597283130317) q[17];
ry(1.5965387947257792) q[19];
cx q[17],q[19];
ry(-0.0039353842564922355) q[17];
ry(3.1413635894982574) q[19];
cx q[17],q[19];
ry(-2.3372400036276018) q[0];
ry(3.0466701258298654) q[1];
cx q[0],q[1];
ry(-2.334177688144773) q[0];
ry(0.7081488840949186) q[1];
cx q[0],q[1];
ry(2.4425337826787072) q[2];
ry(2.117694275157386) q[3];
cx q[2],q[3];
ry(1.9271937479718593) q[2];
ry(2.7752700824193814) q[3];
cx q[2],q[3];
ry(2.8363018885634173) q[4];
ry(2.613517983669621) q[5];
cx q[4],q[5];
ry(1.4937718120785197) q[4];
ry(2.869174610342475) q[5];
cx q[4],q[5];
ry(-1.8355839670599823) q[6];
ry(2.129816125199528) q[7];
cx q[6],q[7];
ry(2.148286176203614) q[6];
ry(-2.1713374295630796) q[7];
cx q[6],q[7];
ry(-2.5861745670078338) q[8];
ry(2.897289910386434) q[9];
cx q[8],q[9];
ry(0.6507177658070908) q[8];
ry(2.355089059273251) q[9];
cx q[8],q[9];
ry(0.2849294781367876) q[10];
ry(2.038387228194881) q[11];
cx q[10],q[11];
ry(-2.4405414331011634) q[10];
ry(0.766496142485389) q[11];
cx q[10],q[11];
ry(2.602606830850003) q[12];
ry(1.0884843862060682) q[13];
cx q[12],q[13];
ry(3.1394784111074374) q[12];
ry(-3.0859873397595887) q[13];
cx q[12],q[13];
ry(-0.8366110289562422) q[14];
ry(-1.5687844635155292) q[15];
cx q[14],q[15];
ry(-2.8086861378240373) q[14];
ry(-1.7526444615741414) q[15];
cx q[14],q[15];
ry(-0.36035114182694494) q[16];
ry(-1.52409316027612) q[17];
cx q[16],q[17];
ry(-0.22096635450315194) q[16];
ry(3.118529189390917) q[17];
cx q[16],q[17];
ry(-1.1167645276082545) q[18];
ry(-2.366221542418173) q[19];
cx q[18],q[19];
ry(2.75170499328896) q[18];
ry(0.051728123829958075) q[19];
cx q[18],q[19];
ry(-1.1374913416490937) q[0];
ry(0.5299724876333152) q[2];
cx q[0],q[2];
ry(0.24719725023295794) q[0];
ry(0.1793038069833015) q[2];
cx q[0],q[2];
ry(-0.14535086388570106) q[2];
ry(-0.21203410395676306) q[4];
cx q[2],q[4];
ry(0.0011682481280711446) q[2];
ry(-0.0002838655346881882) q[4];
cx q[2],q[4];
ry(2.635351148969134) q[4];
ry(-1.750578190638814) q[6];
cx q[4],q[6];
ry(-1.1649917400598477) q[4];
ry(1.52755625372455) q[6];
cx q[4],q[6];
ry(1.519777424467482) q[6];
ry(1.5788169686615845) q[8];
cx q[6],q[8];
ry(-0.06555363619491317) q[6];
ry(0.06485690003502183) q[8];
cx q[6],q[8];
ry(-1.74828017679949) q[8];
ry(2.99034874725947) q[10];
cx q[8],q[10];
ry(3.1363459624454006) q[8];
ry(-0.014385022653028656) q[10];
cx q[8],q[10];
ry(-2.2844734680931986) q[10];
ry(-1.9784598981565784) q[12];
cx q[10],q[12];
ry(3.0727863726974523) q[10];
ry(-0.16255549813670012) q[12];
cx q[10],q[12];
ry(1.0159032916674695) q[12];
ry(-1.1742855148215579) q[14];
cx q[12],q[14];
ry(3.139923872365517) q[12];
ry(-3.1371954855391286) q[14];
cx q[12],q[14];
ry(0.5275418305806223) q[14];
ry(-3.133336032482927) q[16];
cx q[14],q[16];
ry(0.3749507202307417) q[14];
ry(-2.8432784211257154) q[16];
cx q[14],q[16];
ry(-1.4701127026518215) q[16];
ry(-2.3093140410920174) q[18];
cx q[16],q[18];
ry(-3.0358955035379585) q[16];
ry(2.3539133626161552) q[18];
cx q[16],q[18];
ry(2.927391516624393) q[1];
ry(-0.5837479272382815) q[3];
cx q[1],q[3];
ry(0.2873196540891927) q[1];
ry(2.079019328080852) q[3];
cx q[1],q[3];
ry(2.986079817331099) q[3];
ry(2.38969990913395) q[5];
cx q[3],q[5];
ry(3.1403925009501372) q[3];
ry(-0.000544844105575315) q[5];
cx q[3],q[5];
ry(2.369714942163506) q[5];
ry(0.694278392259497) q[7];
cx q[5],q[7];
ry(3.1309889920326137) q[5];
ry(0.4207786236145026) q[7];
cx q[5],q[7];
ry(0.18918486800523612) q[7];
ry(-2.048988550595368) q[9];
cx q[7],q[9];
ry(-2.568027094239327) q[7];
ry(0.02118157578415625) q[9];
cx q[7],q[9];
ry(1.717174599610056) q[9];
ry(1.678920066101165) q[11];
cx q[9],q[11];
ry(2.7264083582091607) q[9];
ry(-0.0016693803901735472) q[11];
cx q[9],q[11];
ry(1.9515479453878677) q[11];
ry(-0.7155108351304423) q[13];
cx q[11],q[13];
ry(-3.141188318172522) q[11];
ry(0.0021290261144608635) q[13];
cx q[11],q[13];
ry(-0.984384708497382) q[13];
ry(1.8383370125041951) q[15];
cx q[13],q[15];
ry(1.3668108511301322) q[13];
ry(0.025415139541204314) q[15];
cx q[13],q[15];
ry(-0.8182671420997147) q[15];
ry(1.2364810893521436) q[17];
cx q[15],q[17];
ry(1.2474335271729886) q[15];
ry(-1.220407538539459) q[17];
cx q[15],q[17];
ry(2.4678589669860385) q[17];
ry(-0.7801696065346254) q[19];
cx q[17],q[19];
ry(1.5174029993640088) q[17];
ry(-0.010025454875397043) q[19];
cx q[17],q[19];
ry(-0.29315332130257943) q[0];
ry(-2.934316728905732) q[1];
cx q[0],q[1];
ry(3.001737287119135) q[0];
ry(0.0970100159647642) q[1];
cx q[0],q[1];
ry(0.5599162471810049) q[2];
ry(0.8041649510241488) q[3];
cx q[2],q[3];
ry(-0.582020921344072) q[2];
ry(0.058378499319411366) q[3];
cx q[2],q[3];
ry(0.16331894087980015) q[4];
ry(1.8521803687155216) q[5];
cx q[4],q[5];
ry(2.086211331231347) q[4];
ry(1.4610651802615602) q[5];
cx q[4],q[5];
ry(-1.6985224774796264) q[6];
ry(1.8591784191261098) q[7];
cx q[6],q[7];
ry(0.10560332279050988) q[6];
ry(1.7571279116855034) q[7];
cx q[6],q[7];
ry(-0.9709912009275523) q[8];
ry(1.1241619213952054) q[9];
cx q[8],q[9];
ry(-0.0024005492113851157) q[8];
ry(0.010999423323702295) q[9];
cx q[8],q[9];
ry(-1.4244802337001445) q[10];
ry(-0.08603140855685835) q[11];
cx q[10],q[11];
ry(-0.9021866620031904) q[10];
ry(-1.6771791264963598) q[11];
cx q[10],q[11];
ry(-1.2312751691891934) q[12];
ry(-2.3702088948212734) q[13];
cx q[12],q[13];
ry(-3.1387345482043765) q[12];
ry(0.028744609208334104) q[13];
cx q[12],q[13];
ry(1.448080439780982) q[14];
ry(-0.3299212259263422) q[15];
cx q[14],q[15];
ry(-0.588940068799249) q[14];
ry(3.080690275566796) q[15];
cx q[14],q[15];
ry(-0.04023711497838165) q[16];
ry(1.1836258351088036) q[17];
cx q[16],q[17];
ry(0.0038300852521881974) q[16];
ry(-2.418724572130687) q[17];
cx q[16],q[17];
ry(1.743219440106503) q[18];
ry(-3.1225568539236406) q[19];
cx q[18],q[19];
ry(0.28736026108875207) q[18];
ry(-0.6189484259534677) q[19];
cx q[18],q[19];
ry(2.5998359314602317) q[0];
ry(2.003901325189853) q[2];
cx q[0],q[2];
ry(3.1064942896791474) q[0];
ry(-1.7185350486603357) q[2];
cx q[0],q[2];
ry(-3.0841301376296655) q[2];
ry(-0.4566567431047081) q[4];
cx q[2],q[4];
ry(-0.005501630381252507) q[2];
ry(0.0006608262371736728) q[4];
cx q[2],q[4];
ry(-0.3184388023730054) q[4];
ry(-1.0295710411312484) q[6];
cx q[4],q[6];
ry(-1.2582186319800628) q[4];
ry(0.44776088923167906) q[6];
cx q[4],q[6];
ry(-0.90106263629393) q[6];
ry(0.5976334788212206) q[8];
cx q[6],q[8];
ry(3.1401086709332495) q[6];
ry(3.1119499271928857) q[8];
cx q[6],q[8];
ry(0.3008749598703311) q[8];
ry(1.5158987095794059) q[10];
cx q[8],q[10];
ry(0.2802679434552494) q[8];
ry(3.0115619000053004) q[10];
cx q[8],q[10];
ry(-1.3176005829350022) q[10];
ry(2.412200539714076) q[12];
cx q[10],q[12];
ry(-3.1359004834446984) q[10];
ry(-0.035774362986558485) q[12];
cx q[10],q[12];
ry(1.649068782814217) q[12];
ry(1.6335952149317103) q[14];
cx q[12],q[14];
ry(-2.841451254196736) q[12];
ry(-0.03731970795350008) q[14];
cx q[12],q[14];
ry(-0.38759474679552675) q[14];
ry(-1.1088186938632978) q[16];
cx q[14],q[16];
ry(-0.0017717046299141614) q[14];
ry(3.1412885453665815) q[16];
cx q[14],q[16];
ry(0.09389397061640885) q[16];
ry(1.4270171010912094) q[18];
cx q[16],q[18];
ry(-1.8932219809421522) q[16];
ry(1.1327472014937099) q[18];
cx q[16],q[18];
ry(-3.078184959751014) q[1];
ry(-0.10522584765668343) q[3];
cx q[1],q[3];
ry(0.26791470541668355) q[1];
ry(-0.7332416405651836) q[3];
cx q[1],q[3];
ry(-1.8244431525397404) q[3];
ry(1.8985122431320534) q[5];
cx q[3],q[5];
ry(-0.26703239356933134) q[3];
ry(3.13652652266128) q[5];
cx q[3],q[5];
ry(1.5717448014801196) q[5];
ry(2.121616626809134) q[7];
cx q[5],q[7];
ry(-3.133985790610429) q[5];
ry(-2.3162270751167995) q[7];
cx q[5],q[7];
ry(-2.6416760092087843) q[7];
ry(-0.8754968488449606) q[9];
cx q[7],q[9];
ry(-0.05138347895301453) q[7];
ry(-2.647218463300484) q[9];
cx q[7],q[9];
ry(-1.0304517351809652) q[9];
ry(2.776836294086691) q[11];
cx q[9],q[11];
ry(-2.763293915637597) q[9];
ry(0.020595531265671596) q[11];
cx q[9],q[11];
ry(-2.2481962220729885) q[11];
ry(-2.2169905562163246) q[13];
cx q[11],q[13];
ry(-3.141512820270097) q[11];
ry(-0.005111552502262917) q[13];
cx q[11],q[13];
ry(-0.2155648916403221) q[13];
ry(1.53748006354195) q[15];
cx q[13],q[15];
ry(1.2944752667438149) q[13];
ry(-2.23545493665366) q[15];
cx q[13],q[15];
ry(0.9847271062780703) q[15];
ry(2.022811188680439) q[17];
cx q[15],q[17];
ry(0.0038923204276364847) q[15];
ry(-3.024909311592098) q[17];
cx q[15],q[17];
ry(0.36093884759277667) q[17];
ry(-0.1691004796475752) q[19];
cx q[17],q[19];
ry(1.4933418744394633) q[17];
ry(-0.007279351122248309) q[19];
cx q[17],q[19];
ry(-2.564583651749034) q[0];
ry(-1.889188424337039) q[1];
cx q[0],q[1];
ry(0.12521101308206417) q[0];
ry(2.44627749535122) q[1];
cx q[0],q[1];
ry(0.7019295856833452) q[2];
ry(1.4109865452524375) q[3];
cx q[2],q[3];
ry(-2.80445056168216) q[2];
ry(2.918781526422836) q[3];
cx q[2],q[3];
ry(1.8796431892796701) q[4];
ry(3.139335085800468) q[5];
cx q[4],q[5];
ry(-1.7348157480555968) q[4];
ry(1.56615034638628) q[5];
cx q[4],q[5];
ry(-0.41247042875771595) q[6];
ry(-1.300594298053754) q[7];
cx q[6],q[7];
ry(2.756577520378123) q[6];
ry(0.12851132223450268) q[7];
cx q[6],q[7];
ry(2.778742786306353) q[8];
ry(-2.8741195357933433) q[9];
cx q[8],q[9];
ry(2.4261211876144158) q[8];
ry(-2.3964491154981755) q[9];
cx q[8],q[9];
ry(2.1027090106547917) q[10];
ry(-0.7496039928181589) q[11];
cx q[10],q[11];
ry(-0.1332625277749102) q[10];
ry(0.29896315222352715) q[11];
cx q[10],q[11];
ry(0.473428051408348) q[12];
ry(-1.6427646547176715) q[13];
cx q[12],q[13];
ry(-2.914456061244682) q[12];
ry(0.006128685644878118) q[13];
cx q[12],q[13];
ry(-0.0645961241718611) q[14];
ry(1.3454512917388826) q[15];
cx q[14],q[15];
ry(0.0009400858087618146) q[14];
ry(0.024335261199196268) q[15];
cx q[14],q[15];
ry(1.2855404306001335) q[16];
ry(-2.7654624888349173) q[17];
cx q[16],q[17];
ry(3.0670285139350257) q[16];
ry(-3.105720645066064) q[17];
cx q[16],q[17];
ry(-1.8287154590802375) q[18];
ry(1.1323815642349362) q[19];
cx q[18],q[19];
ry(-2.6286562226584005) q[18];
ry(-0.014970769261239525) q[19];
cx q[18],q[19];
ry(-0.16951690270665765) q[0];
ry(-2.2880904073496957) q[2];
cx q[0],q[2];
ry(0.20336379058079235) q[0];
ry(1.0372846298444207) q[2];
cx q[0],q[2];
ry(-0.8946443629745096) q[2];
ry(3.0556293121058857) q[4];
cx q[2],q[4];
ry(-0.01598593698494799) q[2];
ry(3.140791810822098) q[4];
cx q[2],q[4];
ry(-1.4905616702600692) q[4];
ry(1.3273918891454068) q[6];
cx q[4],q[6];
ry(-0.032180330341509675) q[4];
ry(0.0954325388349777) q[6];
cx q[4],q[6];
ry(-1.070590325127914) q[6];
ry(-2.9443422299268915) q[8];
cx q[6],q[8];
ry(3.136215932081845) q[6];
ry(3.1250614086529156) q[8];
cx q[6],q[8];
ry(-1.5498294783486237) q[8];
ry(0.524226285372509) q[10];
cx q[8],q[10];
ry(2.831543263048386) q[8];
ry(3.0214378453646726) q[10];
cx q[8],q[10];
ry(-1.636160361544417) q[10];
ry(-2.7656546817156005) q[12];
cx q[10],q[12];
ry(-3.1382643268755857) q[10];
ry(3.0518955307665765) q[12];
cx q[10],q[12];
ry(-2.2131263199908044) q[12];
ry(-0.13034603777027431) q[14];
cx q[12],q[14];
ry(2.684156753546401) q[12];
ry(-0.19315431736170208) q[14];
cx q[12],q[14];
ry(-2.0910977289564747) q[14];
ry(-0.9185294795296994) q[16];
cx q[14],q[16];
ry(3.073292493849069) q[14];
ry(2.9654823455224464) q[16];
cx q[14],q[16];
ry(-1.5818358568633863) q[16];
ry(1.2342794524159442) q[18];
cx q[16],q[18];
ry(3.114576185621135) q[16];
ry(-0.11588501002302641) q[18];
cx q[16],q[18];
ry(1.361667036535539) q[1];
ry(0.007654573930284236) q[3];
cx q[1],q[3];
ry(0.16496135925505223) q[1];
ry(0.05250114322410848) q[3];
cx q[1],q[3];
ry(-3.0868225947905716) q[3];
ry(-2.8947320126637663) q[5];
cx q[3],q[5];
ry(-0.014237844500853747) q[3];
ry(1.9927395737084908e-05) q[5];
cx q[3],q[5];
ry(0.7596072585846896) q[5];
ry(0.19494724989945933) q[7];
cx q[5],q[7];
ry(-0.043767093170106436) q[5];
ry(-0.013889458962397338) q[7];
cx q[5],q[7];
ry(-2.912407799987321) q[7];
ry(0.11150694333216571) q[9];
cx q[7],q[9];
ry(0.002476404953995002) q[7];
ry(0.0038483811825962277) q[9];
cx q[7],q[9];
ry(-1.921123263739092) q[9];
ry(-0.05909154826902867) q[11];
cx q[9],q[11];
ry(0.010504850091260565) q[9];
ry(3.110633848296643) q[11];
cx q[9],q[11];
ry(-1.3778887562944988) q[11];
ry(-0.10400047201789948) q[13];
cx q[11],q[13];
ry(3.076285628121509) q[11];
ry(-0.16521274662034902) q[13];
cx q[11],q[13];
ry(-1.5436000786581845) q[13];
ry(-0.35633124308114095) q[15];
cx q[13],q[15];
ry(3.1091093418470668) q[13];
ry(-3.000914550815031) q[15];
cx q[13],q[15];
ry(-0.5612564587294236) q[15];
ry(-1.573970610542177) q[17];
cx q[15],q[17];
ry(-2.5060138575348754) q[15];
ry(-2.9662610263983393) q[17];
cx q[15],q[17];
ry(-1.2923299052564232) q[17];
ry(0.8064200532404487) q[19];
cx q[17],q[19];
ry(-3.0566735968963052) q[17];
ry(2.9776196097545817) q[19];
cx q[17],q[19];
ry(-1.4590616024333327) q[0];
ry(2.9348808998379186) q[1];
ry(0.6762417081280985) q[2];
ry(1.5234198559214867) q[3];
ry(-3.134452233864099) q[4];
ry(-2.690090988331997) q[5];
ry(-2.8084640910053547) q[6];
ry(-2.0311095987380705) q[7];
ry(0.32895014631411384) q[8];
ry(-0.30163772240974057) q[9];
ry(-3.092734533495354) q[10];
ry(0.013820138695938233) q[11];
ry(-0.013845436885982387) q[12];
ry(-3.121178055964254) q[13];
ry(3.129626353207962) q[14];
ry(-2.7745121682608316) q[15];
ry(-3.0797080544185635) q[16];
ry(-3.1242259211773424) q[17];
ry(1.7931058020435326) q[18];
ry(1.1866548460256618) q[19];