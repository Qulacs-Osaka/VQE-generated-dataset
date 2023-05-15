OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(2.2938868874046188) q[0];
rz(0.6381075600209004) q[0];
ry(2.9117989262050616) q[1];
rz(-1.7953531925903272) q[1];
ry(2.4635029772136234) q[2];
rz(2.860944469114232) q[2];
ry(-3.141564580766741) q[3];
rz(-0.24945675669411482) q[3];
ry(-1.2358291855456116) q[4];
rz(-0.10726753387396548) q[4];
ry(-0.024572187315151872) q[5];
rz(-3.015570190333455) q[5];
ry(8.410337316042742e-06) q[6];
rz(-0.4893980070641263) q[6];
ry(1.5600964109052156) q[7];
rz(1.0995798146139304) q[7];
ry(-1.5574299063789305) q[8];
rz(-0.034052755244216464) q[8];
ry(-0.0005121421818863325) q[9];
rz(2.945275083799461) q[9];
ry(0.9956257582433894) q[10];
rz(-0.703078922256001) q[10];
ry(0.010088462210739939) q[11];
rz(-0.41731890529525906) q[11];
ry(3.0782949104890034) q[12];
rz(-0.07496700780322728) q[12];
ry(-1.8233165377856042) q[13];
rz(-1.7907740892060415) q[13];
ry(3.040531470875601) q[14];
rz(-2.043367118948864) q[14];
ry(-0.04790166658267747) q[15];
rz(-2.1189593673666742) q[15];
ry(-1.5709153338061905) q[16];
rz(-1.6310714433523077) q[16];
ry(0.33901749014521343) q[17];
rz(1.942095372572104) q[17];
ry(-2.6234676433667024) q[18];
rz(0.3947911904988093) q[18];
ry(1.5149649097638118) q[19];
rz(2.4958056551354595) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(2.7112956713735556) q[0];
rz(1.2496099504128493) q[0];
ry(1.573187033576863) q[1];
rz(2.173475179708567) q[1];
ry(-0.7596422777346657) q[2];
rz(-2.8880302690362787) q[2];
ry(-1.9518081036394728) q[3];
rz(1.456151491623217) q[3];
ry(1.815380742054196) q[4];
rz(-3.0420603237152415) q[4];
ry(-0.009095287014378648) q[5];
rz(0.7385532875870515) q[5];
ry(-1.5708089604128037) q[6];
rz(-0.7093147615011316) q[6];
ry(1.0508182508535509) q[7];
rz(0.6679153268581801) q[7];
ry(-2.4971078729740306) q[8];
rz(-0.6945127710883074) q[8];
ry(-3.1238113264483807) q[9];
rz(-2.5902700352501187) q[9];
ry(-2.3948574530327056) q[10];
rz(-1.8231276203667397) q[10];
ry(2.0702347606266316) q[11];
rz(0.7863209193882609) q[11];
ry(-0.8251219220407755) q[12];
rz(0.2350558676039105) q[12];
ry(-1.6407549444288572) q[13];
rz(1.0522080968125014) q[13];
ry(3.1361511727035634) q[14];
rz(1.2017475396327784) q[14];
ry(-0.003008060918111762) q[15];
rz(-0.13791307913709347) q[15];
ry(2.5262350045094832) q[16];
rz(-0.5095819643466406) q[16];
ry(-0.2009724294552192) q[17];
rz(-1.5304702893751936) q[17];
ry(0.38914415962745696) q[18];
rz(1.9265181515016465) q[18];
ry(0.5105971787672102) q[19];
rz(-1.9360207696183074) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(2.96346735152007) q[0];
rz(0.8467810631851689) q[0];
ry(2.917668573013994) q[1];
rz(-0.6931179201460451) q[1];
ry(0.0002943432337402841) q[2];
rz(2.9460286552403354) q[2];
ry(3.1393814520961305) q[3];
rz(1.4566683185544747) q[3];
ry(-3.1407579602954527) q[4];
rz(2.4099117673446804) q[4];
ry(-1.5707791657609398) q[5];
rz(0.4121871488706875) q[5];
ry(-0.21861134101030455) q[6];
rz(-0.01376375598684866) q[6];
ry(-2.4980603317546097) q[7];
rz(-0.2352588584588311) q[7];
ry(0.7270993160651011) q[8];
rz(2.441965797789724) q[8];
ry(3.138317997998006) q[9];
rz(-2.4609963735350977) q[9];
ry(2.724618915493718) q[10];
rz(-0.04008223557249923) q[10];
ry(0.0001423389197983127) q[11];
rz(-0.5842575643287844) q[11];
ry(1.3090973190513253) q[12];
rz(0.0021746969146647643) q[12];
ry(-2.4132419028294025) q[13];
rz(-1.2307455212447183) q[13];
ry(0.21025252510384107) q[14];
rz(0.46824878589185626) q[14];
ry(3.0923725202247168) q[15];
rz(0.05275382487639849) q[15];
ry(-1.8419852200168334) q[16];
rz(-1.5231828373047416) q[16];
ry(-2.804469802843714) q[17];
rz(-2.241032880110155) q[17];
ry(-2.4645054081124647) q[18];
rz(2.6141735833933706) q[18];
ry(2.026454020596053) q[19];
rz(0.29835254379386544) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(1.8554290081340048) q[0];
rz(-2.6962536405950557) q[0];
ry(-1.6295928875033985) q[1];
rz(-1.3550719772787385) q[1];
ry(-2.337013759950385) q[2];
rz(2.9968139551602535) q[2];
ry(-1.9516080035620131) q[3];
rz(-2.5634187956071752) q[3];
ry(-1.5707788892427283) q[4];
rz(-1.7606347387400074) q[4];
ry(0.42509764175986675) q[5];
rz(3.0778607472885917) q[5];
ry(1.4648029174824257) q[6];
rz(2.68742476100354) q[6];
ry(-0.36296088832037476) q[7];
rz(-1.1636761526431147) q[7];
ry(-2.8462860949085047) q[8];
rz(0.05491096787578531) q[8];
ry(1.6802079518664055) q[9];
rz(-0.5949985759592611) q[9];
ry(-2.7728944863444154) q[10];
rz(1.2583752216862676) q[10];
ry(3.137480372713179) q[11];
rz(1.5517738913858956) q[11];
ry(-1.028346543590575) q[12];
rz(-3.1410710354808775) q[12];
ry(-0.00037843160491914774) q[13];
rz(1.7360729519998623) q[13];
ry(0.008024418219214624) q[14];
rz(3.096022433141242) q[14];
ry(3.133038802878764) q[15];
rz(2.62552036597841) q[15];
ry(2.938727668048659) q[16];
rz(0.13599145765127396) q[16];
ry(2.991701390436962) q[17];
rz(2.2102045094011125) q[17];
ry(-3.0587916231165235) q[18];
rz(2.3402182679359806) q[18];
ry(1.7574480484808606) q[19];
rz(-1.273427627433867) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-0.8495176284354381) q[0];
rz(-2.3858491443198595) q[0];
ry(-0.18574893812149895) q[1];
rz(2.6388803768595475) q[1];
ry(1.5424320839879755) q[2];
rz(0.1355030211532322) q[2];
ry(-1.5704769710896729) q[3];
rz(0.34698545520469504) q[3];
ry(-2.44294331233203) q[4];
rz(-1.0644927246318474) q[4];
ry(0.1156304752268417) q[5];
rz(-0.700113627731815) q[5];
ry(-3.0391987539802687) q[6];
rz(2.848985609667253) q[6];
ry(-0.008457356053726883) q[7];
rz(-1.1693578577211179) q[7];
ry(0.0008698625187895105) q[8];
rz(-2.704734415873119) q[8];
ry(-3.140627837997465) q[9];
rz(-2.082237729870144) q[9];
ry(0.8135234725120684) q[10];
rz(-0.8144452021246957) q[10];
ry(0.0001820468426450006) q[11];
rz(-0.950502878645599) q[11];
ry(1.8310227347647237) q[12];
rz(-1.6648726399228266) q[12];
ry(-1.7449501402896062) q[13];
rz(1.2436940742326033) q[13];
ry(2.9075026697878776) q[14];
rz(1.983006304498346) q[14];
ry(3.1271232861204963) q[15];
rz(-0.32178507824809527) q[15];
ry(2.538546917491559) q[16];
rz(-2.9448029298703053) q[16];
ry(1.7664061247164546) q[17];
rz(1.8179949697432765) q[17];
ry(-2.803287326682987) q[18];
rz(-2.206710825082701) q[18];
ry(3.1313279273094707) q[19];
rz(2.154108106578905) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(2.8150589801859645) q[0];
rz(0.44365352389725204) q[0];
ry(-0.10573346743154134) q[1];
rz(0.6984178438040068) q[1];
ry(-1.5732875123159662) q[2];
rz(-3.085205510277692) q[2];
ry(3.1415344616918692) q[3];
rz(1.897561054997562) q[3];
ry(0.004287941531424925) q[4];
rz(-1.8757163404853607) q[4];
ry(3.02112387576329) q[5];
rz(0.09504062856861421) q[5];
ry(-1.9894820097011374) q[6];
rz(0.6191372854615548) q[6];
ry(0.9009065884979117) q[7];
rz(1.002928312053378) q[7];
ry(0.43236353262693694) q[8];
rz(-0.2933832469547193) q[8];
ry(2.250619617386503) q[9];
rz(-2.1847564004156084) q[9];
ry(0.7894139801394973) q[10];
rz(0.9159851401662121) q[10];
ry(1.209392233006736) q[11];
rz(-1.3844846999648608) q[11];
ry(0.059688102011766865) q[12];
rz(-1.2349161999795144) q[12];
ry(1.8447612675113594) q[13];
rz(-2.5134318767598907) q[13];
ry(0.09625367532328853) q[14];
rz(2.5420107283686635) q[14];
ry(3.11646822918389) q[15];
rz(1.3528945986230356) q[15];
ry(0.3306164911821533) q[16];
rz(0.2348043537880802) q[16];
ry(0.8532879810839383) q[17];
rz(-1.2605284356254944) q[17];
ry(-0.14514488973390982) q[18];
rz(0.5394801018100929) q[18];
ry(-2.0561103833931207) q[19];
rz(2.229953842716842) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-1.458479150579229) q[0];
rz(-3.052619587017558) q[0];
ry(0.31685965031980856) q[1];
rz(-2.892158540823628) q[1];
ry(-1.526728323300138) q[2];
rz(-1.7313442566517239) q[2];
ry(1.9498598652011347) q[3];
rz(-2.196931210470212) q[3];
ry(2.848721270256314) q[4];
rz(-2.3597463891956116) q[4];
ry(2.5335035892490763) q[5];
rz(-2.905125657302866) q[5];
ry(2.9185639723386347) q[6];
rz(-2.9610282846519715) q[6];
ry(1.0901386920932028) q[7];
rz(0.12952015698898608) q[7];
ry(2.248204003118679) q[8];
rz(-1.4532948141295856) q[8];
ry(-0.2116663286940048) q[9];
rz(-0.21088524483703883) q[9];
ry(-1.2299910784849075) q[10];
rz(-0.07882174690478304) q[10];
ry(3.1415713727800147) q[11];
rz(1.2711620436729119) q[11];
ry(0.0034377201461453536) q[12];
rz(0.05242464225777603) q[12];
ry(2.902839694609624) q[13];
rz(-0.4322177881363308) q[13];
ry(1.5759780648246995) q[14];
rz(1.2211358585023744) q[14];
ry(-0.9076945532506979) q[15];
rz(1.5545997224889527) q[15];
ry(-0.19458463439705476) q[16];
rz(1.7521314891938766) q[16];
ry(1.1671072440020385) q[17];
rz(-0.6707127054981611) q[17];
ry(-3.138367382650643) q[18];
rz(1.378777161551742) q[18];
ry(-1.6815512455511152) q[19];
rz(1.244948488913657) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(2.9833952095125067) q[0];
rz(3.0022920822998613) q[0];
ry(-2.7581622998413398) q[1];
rz(-2.560726291971411) q[1];
ry(-0.006607376465405355) q[2];
rz(1.4111947891660042) q[2];
ry(-0.002799879079269837) q[3];
rz(-0.9176447145543439) q[3];
ry(2.345970322431804) q[4];
rz(0.04018489483555409) q[4];
ry(1.5040489566596023) q[5];
rz(-1.2278599328495021) q[5];
ry(-3.0042826766001016) q[6];
rz(-1.3142317665013072) q[6];
ry(-0.0035311625534735214) q[7];
rz(0.4534540397959326) q[7];
ry(-3.139946640647217) q[8];
rz(1.4194550307989973) q[8];
ry(-0.7608533135315049) q[9];
rz(0.8158198791305018) q[9];
ry(0.16767740899132288) q[10];
rz(-0.452323642853579) q[10];
ry(2.3002060565000932) q[11];
rz(-0.44003490202491335) q[11];
ry(-0.7654399258916346) q[12];
rz(-1.756718528464541) q[12];
ry(-0.06187243631114647) q[13];
rz(1.2705853339708355) q[13];
ry(-3.140923296935063) q[14];
rz(-1.846847962580456) q[14];
ry(-0.00027452676130224063) q[15];
rz(-1.5468661019562993) q[15];
ry(3.1111967050620914) q[16];
rz(0.9052601110670371) q[16];
ry(2.336086360528682) q[17];
rz(-0.7470614507332788) q[17];
ry(-0.463371573276632) q[18];
rz(0.17533210275771224) q[18];
ry(0.21042423047251457) q[19];
rz(0.38989249646896257) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-2.280330822834549) q[0];
rz(-0.2375967529738263) q[0];
ry(1.8377913517154902) q[1];
rz(1.6141943227514206) q[1];
ry(-2.693397676063385) q[2];
rz(2.622715527273672) q[2];
ry(-0.1555938236818024) q[3];
rz(-1.3916028518065393) q[3];
ry(-1.416001965197096) q[4];
rz(0.7597921201720091) q[4];
ry(-2.19848988710508) q[5];
rz(-0.02360551492817543) q[5];
ry(-0.048726648369696875) q[6];
rz(-1.8326219957332175) q[6];
ry(0.9654355235816505) q[7];
rz(-3.0194487742552534) q[7];
ry(3.0179944435973596) q[8];
rz(-0.8563635207712931) q[8];
ry(2.1749132689748465) q[9];
rz(-0.27405810610387693) q[9];
ry(-0.014871703076005716) q[10];
rz(0.7781988746422288) q[10];
ry(-0.25137757111083564) q[11];
rz(-2.4814941334738854) q[11];
ry(-1.5707777446561748) q[12];
rz(-1.621684971428862) q[12];
ry(-2.1502049952195215) q[13];
rz(-2.6514341776042403) q[13];
ry(-1.594911380783671) q[14];
rz(-2.0672629235663615) q[14];
ry(0.9383690667050294) q[15];
rz(0.34853766093318495) q[15];
ry(-0.32787806511882955) q[16];
rz(0.9899109178044259) q[16];
ry(-0.6571367038458985) q[17];
rz(0.6791343853658631) q[17];
ry(0.054739356866173046) q[18];
rz(-2.345778461969586) q[18];
ry(2.358229047671121) q[19];
rz(1.529114534282735) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(1.789143688864989) q[0];
rz(-2.9424851096062477) q[0];
ry(1.4280951780833009) q[1];
rz(2.9748393541495903) q[1];
ry(-3.1123799921647506) q[2];
rz(-0.1373069187884548) q[2];
ry(-0.00029886583234262076) q[3];
rz(2.806195343191161) q[3];
ry(3.0041837418027164) q[4];
rz(0.898429191349443) q[4];
ry(2.737585336955068) q[5];
rz(0.008874490889525077) q[5];
ry(-3.087309622557385) q[6];
rz(-0.041102124621184684) q[6];
ry(-3.134839249455806) q[7];
rz(1.2906254732106408) q[7];
ry(0.009800769782589923) q[8];
rz(-0.32005879625232553) q[8];
ry(-1.1451607226083595) q[9];
rz(-2.0711116341037865) q[9];
ry(3.109799305516901) q[10];
rz(-2.6067974107113403) q[10];
ry(0.2481270486955045) q[11];
rz(0.4305395859696224) q[11];
ry(-0.026867602903292334) q[12];
rz(3.140190724174002) q[12];
ry(-1.5707949749393055) q[13];
rz(-2.5802094869408623) q[13];
ry(-0.021852189552129364) q[14];
rz(1.569880287722342) q[14];
ry(0.01869920562905375) q[15];
rz(2.587890307159048) q[15];
ry(-0.5096363378137613) q[16];
rz(0.3578683219672356) q[16];
ry(-2.8148250842181963) q[17];
rz(1.0407000390304468) q[17];
ry(-2.9666072668798953) q[18];
rz(0.6500513684022192) q[18];
ry(1.347603999232905) q[19];
rz(-1.1042647142137323) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-0.3231528902918825) q[0];
rz(-3.1342560238949324) q[0];
ry(2.889237881314811) q[1];
rz(-1.2386325773704323) q[1];
ry(-1.7290552468318663) q[2];
rz(-1.1704858211031723) q[2];
ry(3.0795831499613873) q[3];
rz(-1.91005270401064) q[3];
ry(0.8970677714656796) q[4];
rz(2.734460770588608) q[4];
ry(-0.6233384608244236) q[5];
rz(1.8649324498316853) q[5];
ry(-0.4209276581210819) q[6];
rz(-0.46019308416744686) q[6];
ry(0.543151613757761) q[7];
rz(-1.4254277122729748) q[7];
ry(-2.34618812087948) q[8];
rz(1.535483905085442) q[8];
ry(-2.2277759201766276) q[9];
rz(2.361016349688901) q[9];
ry(-0.0028170443526792286) q[10];
rz(2.3135272670777822) q[10];
ry(-2.8674149282171926) q[11];
rz(-0.8048629611880008) q[11];
ry(-2.6506860565015646) q[12];
rz(-1.6517016799941588) q[12];
ry(2.3361806422636446) q[13];
rz(2.3954984661597845) q[13];
ry(1.5708690755234365) q[14];
rz(-2.971055099552463) q[14];
ry(-3.0549592386130975) q[15];
rz(0.1059638674935984) q[15];
ry(1.6896526277908404) q[16];
rz(-3.0317746508020442) q[16];
ry(-1.5616909652652686) q[17];
rz(-1.1480232558279677) q[17];
ry(-1.7688419676687077) q[18];
rz(1.7778725542830276) q[18];
ry(-0.6812560598118491) q[19];
rz(2.545635114012095) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-2.8450070226387396) q[0];
rz(0.21633613922147263) q[0];
ry(-0.5417790327161542) q[1];
rz(-1.4992763255144463) q[1];
ry(0.006586787950970739) q[2];
rz(1.3187567396871769) q[2];
ry(3.0951306624043227) q[3];
rz(-3.0523769533438827) q[3];
ry(0.1959170839599194) q[4];
rz(2.529866972235921) q[4];
ry(3.0021268124446134) q[5];
rz(-0.8740441225198091) q[5];
ry(-0.06619238657351811) q[6];
rz(0.24899539773439303) q[6];
ry(3.1392726634754573) q[7];
rz(0.7544106996279991) q[7];
ry(-2.0479040003268674) q[8];
rz(-3.140664760219851) q[8];
ry(-0.08200765060521154) q[9];
rz(2.4727612890789565) q[9];
ry(-0.04472107797645528) q[10];
rz(2.8433860819432444) q[10];
ry(2.4457094328008795) q[11];
rz(-1.3013593576681712) q[11];
ry(-0.009421984300851172) q[12];
rz(3.0923544255571063) q[12];
ry(3.0888055354272494) q[13];
rz(1.110362045299584) q[13];
ry(2.8956125249366975) q[14];
rz(1.675902395227616) q[14];
ry(1.570796301910462) q[15];
rz(1.0192767355190178) q[15];
ry(-1.123711947498006) q[16];
rz(-0.47328537927221237) q[16];
ry(0.14611993831153214) q[17];
rz(1.0969402261289956) q[17];
ry(1.8729177330804374) q[18];
rz(2.095036296698485) q[18];
ry(0.8069729912238861) q[19];
rz(0.4060503930558397) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.6021434488183469) q[0];
rz(-0.6296927485842223) q[0];
ry(0.8135555804514986) q[1];
rz(-0.8946772092793415) q[1];
ry(0.05456011213155657) q[2];
rz(-0.18595813590072208) q[2];
ry(0.15573685971958007) q[3];
rz(-1.9865540755936095) q[3];
ry(0.9083804679124068) q[4];
rz(0.6623625817571803) q[4];
ry(0.08372390080201715) q[5];
rz(3.0822749010510795) q[5];
ry(0.8103743939690968) q[6];
rz(2.426007164318776) q[6];
ry(-0.06320572076990348) q[7];
rz(-3.1037100282129098) q[7];
ry(2.139690360710521) q[8];
rz(0.016627329724279605) q[8];
ry(3.000995474263773) q[9];
rz(0.07180968169300514) q[9];
ry(0.07707120803671774) q[10];
rz(1.1652998442235523) q[10];
ry(0.5178539733128339) q[11];
rz(-2.43352462184632) q[11];
ry(-2.4696948830624508) q[12];
rz(0.08556732338789794) q[12];
ry(3.058138998530575) q[13];
rz(1.9372664371833825) q[13];
ry(3.1382757291854606) q[14];
rz(-3.0241000115413885) q[14];
ry(0.015543063622264814) q[15];
rz(0.6026091538384692) q[15];
ry(-1.5707926228034346) q[16];
rz(0.4971060265160743) q[16];
ry(2.906138491174036) q[17];
rz(1.2972296584281908) q[17];
ry(-0.18412465031415082) q[18];
rz(-2.987817998886647) q[18];
ry(3.076051539294825) q[19];
rz(0.5527768409132436) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(3.0129878929072538) q[0];
rz(-0.8007275488418824) q[0];
ry(-0.7061053032380018) q[1];
rz(1.7664582245262033) q[1];
ry(2.9817430252120203) q[2];
rz(0.22863046741106086) q[2];
ry(0.08102837726847323) q[3];
rz(2.7699292175235195) q[3];
ry(-1.6749661303735355) q[4];
rz(-2.6829945380211426) q[4];
ry(-0.027311947508363858) q[5];
rz(-1.0703965103321185) q[5];
ry(3.1178412190125684) q[6];
rz(-0.3600541896695644) q[6];
ry(3.137731814597681) q[7];
rz(-0.06159971204553604) q[7];
ry(-1.084235906576591) q[8];
rz(-2.4161494050226455) q[8];
ry(-0.8356039968612352) q[9];
rz(1.3781995412675734) q[9];
ry(3.1396208233451164) q[10];
rz(0.837311558849164) q[10];
ry(-3.131592337322867) q[11];
rz(3.079093381485073) q[11];
ry(-0.010394731326935691) q[12];
rz(-1.669928350741249) q[12];
ry(-0.029177143464080135) q[13];
rz(1.6630433666776883) q[13];
ry(0.059807805095519306) q[14];
rz(1.9284971549666832) q[14];
ry(-1.613468955982774) q[15];
rz(2.962474346754373) q[15];
ry(-0.1687626530663832) q[16];
rz(0.6605499665167696) q[16];
ry(-1.5707961222264863) q[17];
rz(-3.0288257481567036) q[17];
ry(-2.0883723323213395) q[18];
rz(0.3012196207758438) q[18];
ry(2.3923366022701953) q[19];
rz(-0.41055644727847546) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(2.410569992843137) q[0];
rz(-2.5796679591614606) q[0];
ry(0.7676192200034269) q[1];
rz(0.5354594352465482) q[1];
ry(1.8317163428189485) q[2];
rz(3.0966527386220566) q[2];
ry(-0.015101908600624547) q[3];
rz(1.2903935368729367) q[3];
ry(-0.8155212833362431) q[4];
rz(2.245826129960985) q[4];
ry(-3.02638317065382) q[5];
rz(-0.3812028944370106) q[5];
ry(1.9796207196808213) q[6];
rz(-0.07297394619322262) q[6];
ry(1.0827757978077583) q[7];
rz(-1.9654945662101788) q[7];
ry(0.7147849489717222) q[8];
rz(-2.7866692577665746) q[8];
ry(2.967665290117275) q[9];
rz(0.6040886722860002) q[9];
ry(1.6612786141403664) q[10];
rz(2.198830953531835) q[10];
ry(0.6600519921799223) q[11];
rz(0.9825092462324807) q[11];
ry(1.8066748281695064) q[12];
rz(0.914232148621954) q[12];
ry(-0.07144560430322205) q[13];
rz(0.4298762341330404) q[13];
ry(2.3959470902277284) q[14];
rz(0.27866808724608605) q[14];
ry(-0.12885877182032954) q[15];
rz(-2.3638410536117513) q[15];
ry(0.004761106308352581) q[16];
rz(0.9610961440150847) q[16];
ry(3.1358667272375853) q[17];
rz(-3.0329102801695638) q[17];
ry(-1.5707950209831596) q[18];
rz(2.9543444310602824) q[18];
ry(2.943954420791901) q[19];
rz(2.2731166791196635) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-0.03417950008963456) q[0];
rz(2.330495136533043) q[0];
ry(-3.07728983304856) q[1];
rz(-1.3663608431046017) q[1];
ry(2.7717587112357163) q[2];
rz(-1.628756543862515) q[2];
ry(-0.020475346127442097) q[3];
rz(-2.0899304028329384) q[3];
ry(0.7864123328878403) q[4];
rz(-2.0181171077434845) q[4];
ry(-3.1243380689004936) q[5];
rz(-0.21737686771466952) q[5];
ry(-0.016401473896843655) q[6];
rz(0.35233102855330056) q[6];
ry(0.05286635954206531) q[7];
rz(1.6296574635500003) q[7];
ry(0.04059672384996826) q[8];
rz(-2.235966630326052) q[8];
ry(3.114527594480665) q[9];
rz(-1.7200662588792373) q[9];
ry(-0.05247212923776079) q[10];
rz(-2.8179847705762615) q[10];
ry(0.04344904411480787) q[11];
rz(-0.8015429183124687) q[11];
ry(-2.960477316116012) q[12];
rz(-2.117918886323065) q[12];
ry(3.07253699200756) q[13];
rz(3.0814714685812925) q[13];
ry(-2.870600480951714) q[14];
rz(-2.210840599845631) q[14];
ry(3.1303345755602394) q[15];
rz(3.0834725807256986) q[15];
ry(3.0444877159019645) q[16];
rz(1.9001814407611903) q[16];
ry(1.7263670415134733) q[17];
rz(-2.631897475235834) q[17];
ry(-2.4237469285761106) q[18];
rz(2.5905253730638487) q[18];
ry(-1.5707980743824141) q[19];
rz(1.302169755365879) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.6499763331272185) q[0];
rz(-0.34126211738630485) q[0];
ry(-0.22655935979058395) q[1];
rz(1.7655633053183042) q[1];
ry(-1.2682058883702858) q[2];
rz(0.14376940273513306) q[2];
ry(1.9783332578295576) q[3];
rz(-0.8747780111735569) q[3];
ry(-2.720543123336572) q[4];
rz(0.750763301790686) q[4];
ry(-1.186966389018138) q[5];
rz(2.35968867223538) q[5];
ry(0.30363595286585227) q[6];
rz(0.6460384227653542) q[6];
ry(3.0969858612536956) q[7];
rz(1.3808563164648122) q[7];
ry(-1.5791241159569864) q[8];
rz(0.27224048055860006) q[8];
ry(-1.3163275017162162) q[9];
rz(-2.077474306437412) q[9];
ry(0.9033214655809575) q[10];
rz(-2.91269953527052) q[10];
ry(0.8274376942896818) q[11];
rz(-2.695691766848393) q[11];
ry(1.0225302358972164) q[12];
rz(-0.029762538504851457) q[12];
ry(-2.1519930175276727) q[13];
rz(0.10922478066871566) q[13];
ry(1.0228733648411392) q[14];
rz(-1.810713716265143) q[14];
ry(-1.9445730419483211) q[15];
rz(-2.8009513333437424) q[15];
ry(-0.3733759569417909) q[16];
rz(-2.63387824209336) q[16];
ry(0.6548110081002445) q[17];
rz(1.3904845942560398) q[17];
ry(1.7198695643665463) q[18];
rz(-1.4686696549453817) q[18];
ry(-2.7290501953550934) q[19];
rz(-0.1267907566667406) q[19];