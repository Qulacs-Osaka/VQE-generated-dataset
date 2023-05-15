OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(3.122734561567732) q[0];
rz(-1.5654050751735271) q[0];
ry(-1.569027208456824) q[1];
rz(-1.761166858003187) q[1];
ry(-3.0317758471860135) q[2];
rz(0.5420905405427723) q[2];
ry(-1.5639649266351334) q[3];
rz(2.9222293823743097) q[3];
ry(2.2511644227611702) q[4];
rz(-0.05713084828995471) q[4];
ry(-0.028312499671940827) q[5];
rz(0.20090435834322468) q[5];
ry(0.5016222550046843) q[6];
rz(0.042956027708002296) q[6];
ry(-3.1411391647897715) q[7];
rz(0.6660546355369393) q[7];
ry(0.05577932224292059) q[8];
rz(-1.771681095020736) q[8];
ry(-2.363147184499777) q[9];
rz(0.5253219755155936) q[9];
ry(-0.0006275716273736213) q[10];
rz(1.8895305035328898) q[10];
ry(1.5586803142244794) q[11];
rz(3.123828732215603) q[11];
ry(0.44383095371557707) q[12];
rz(-2.857068347077809) q[12];
ry(0.19061661674574104) q[13];
rz(-1.313871311450562) q[13];
ry(0.35612452768955993) q[14];
rz(-0.901341305307369) q[14];
ry(1.5594481671228264) q[15];
rz(0.5709981163256279) q[15];
ry(2.3308592118047646) q[16];
rz(-2.43550403085125) q[16];
ry(-0.0038224826780393073) q[17];
rz(-0.12114784992100411) q[17];
ry(0.5217571536819241) q[18];
rz(-2.587053935905178) q[18];
ry(-0.5991605221351719) q[19];
rz(1.071576772147341) q[19];
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
ry(-1.5720249340018222) q[0];
rz(-0.28770653566394194) q[0];
ry(2.6903040451031526) q[1];
rz(0.5515445038825302) q[1];
ry(0.013663198683646627) q[2];
rz(-2.1327615973481553) q[2];
ry(1.7413489467765624) q[3];
rz(1.4281038250201625) q[3];
ry(3.0278772109566483) q[4];
rz(-1.6036458763411572) q[4];
ry(1.5701639815259556) q[5];
rz(1.614398265241621) q[5];
ry(0.08756396589146485) q[6];
rz(0.5105444338588868) q[6];
ry(0.00018632647935135083) q[7];
rz(2.311654932747611) q[7];
ry(1.5706679428251715) q[8];
rz(1.473173661934438) q[8];
ry(0.016720844218124142) q[9];
rz(-2.9119104703306484) q[9];
ry(-1.6006556231854923) q[10];
rz(0.003057892921324523) q[10];
ry(1.59661359309876) q[11];
rz(-2.6525339708237476) q[11];
ry(-2.5788397483210073) q[12];
rz(0.36793948696109974) q[12];
ry(3.1412741511181137) q[13];
rz(-1.1853330802978743) q[13];
ry(0.005188940406677127) q[14];
rz(0.9421433060328218) q[14];
ry(-3.120276100692376) q[15];
rz(-2.9499699088703784) q[15];
ry(3.1413499031919137) q[16];
rz(-0.43845426487814265) q[16];
ry(3.0082669489240637) q[17];
rz(2.650993137125957) q[17];
ry(-2.9575557427303734) q[18];
rz(-2.463246911417656) q[18];
ry(-0.4787635596963389) q[19];
rz(-0.7693604607554265) q[19];
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
ry(-1.7600711382355172) q[0];
rz(-2.3999647267521826) q[0];
ry(-1.5564330046787964) q[1];
rz(1.559396032289199) q[1];
ry(-1.4622997307747356) q[2];
rz(-3.1367901852120514) q[2];
ry(-2.9237140052222625) q[3];
rz(-0.07013960354747972) q[3];
ry(-0.17719581751801236) q[4];
rz(1.5839928619239743) q[4];
ry(-1.6594360774371277) q[5];
rz(-0.8325843054267166) q[5];
ry(3.133915103810069) q[6];
rz(-1.315099281421859) q[6];
ry(-1.5704145891302872) q[7];
rz(1.6001173890270133) q[7];
ry(0.012304630901553537) q[8];
rz(-2.745491276245133) q[8];
ry(-0.05889010690756903) q[9];
rz(-2.2837946752234353) q[9];
ry(1.5679818478917367) q[10];
rz(-0.8196623846989245) q[10];
ry(-2.559139950663036) q[11];
rz(0.021278688238253483) q[11];
ry(3.134087052909182) q[12];
rz(-1.0854906026824955) q[12];
ry(-0.006338238095572102) q[13];
rz(-0.8467717606552064) q[13];
ry(0.2697655918839986) q[14];
rz(0.8503427032866259) q[14];
ry(3.133198048920413) q[15];
rz(-0.4859819876293931) q[15];
ry(2.1210834816521604) q[16];
rz(3.004561758484264) q[16];
ry(-0.0014997638731785203) q[17];
rz(-0.26101530113644067) q[17];
ry(0.5079218376627899) q[18];
rz(-1.7799666213615637) q[18];
ry(-2.4513100842114746) q[19];
rz(0.6017899881976634) q[19];
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
ry(3.136742131650496) q[0];
rz(3.0256953232038386) q[0];
ry(-1.4699682707761281) q[1];
rz(1.521741967352436) q[1];
ry(-1.7397361465019134) q[2];
rz(1.5779898907792829) q[2];
ry(0.8764406426060516) q[3];
rz(-1.624761720385967) q[3];
ry(3.0719291802140885) q[4];
rz(-0.01629134528758994) q[4];
ry(1.5651331862371896) q[5];
rz(-1.1362329082775111) q[5];
ry(-3.1413259456903857) q[6];
rz(-2.9874395644957823) q[6];
ry(0.07492025758546253) q[7];
rz(3.12212441797618) q[7];
ry(-0.031478513719635615) q[8];
rz(-2.786762403651885) q[8];
ry(3.1397137982916488) q[9];
rz(-0.7681326930852297) q[9];
ry(-3.095937691260292) q[10];
rz(0.5955316154420052) q[10];
ry(0.0028100195147748863) q[11];
rz(3.1152560862047234) q[11];
ry(1.5731910893868002) q[12];
rz(1.3024713871255305) q[12];
ry(0.0003278640868842331) q[13];
rz(-2.4503790203855362) q[13];
ry(-0.005262852069608748) q[14];
rz(-1.4326098527070812) q[14];
ry(-3.1238020447926376) q[15];
rz(1.4937940418229612) q[15];
ry(0.008507454108270807) q[16];
rz(-0.357761874934365) q[16];
ry(2.6788202555636094) q[17];
rz(1.0371322618168) q[17];
ry(1.4873822537873944) q[18];
rz(-1.0673320707498664) q[18];
ry(0.15107778683171968) q[19];
rz(1.312841641384297) q[19];
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
ry(0.937211887702095) q[0];
rz(1.3396251371102066) q[0];
ry(0.7044748667960067) q[1];
rz(1.8213163202906806) q[1];
ry(-1.5655449727989863) q[2];
rz(1.2181411376850573) q[2];
ry(-0.0016728244269517134) q[3];
rz(0.8447072576460449) q[3];
ry(1.5701110936439546) q[4];
rz(-1.6617501139529716) q[4];
ry(1.6861410351331827) q[5];
rz(1.0931553754252221) q[5];
ry(3.1175564326348533) q[6];
rz(0.6936918550429576) q[6];
ry(-1.5552225362249006) q[7];
rz(-0.30961785231418565) q[7];
ry(-1.5630298151396573) q[8];
rz(1.559406295913261) q[8];
ry(-3.0891591884172653) q[9];
rz(2.4991843817112978) q[9];
ry(1.4447092223094606) q[10];
rz(-1.404245181535969) q[10];
ry(0.5850373172454022) q[11];
rz(-2.2245535264144616) q[11];
ry(-1.5304594865471401) q[12];
rz(1.3161699416466632) q[12];
ry(1.5700901641316456) q[13];
rz(0.8979083021758161) q[13];
ry(3.1380906088528615) q[14];
rz(-1.4888976061200232) q[14];
ry(0.20191439268195052) q[15];
rz(3.105521655336984) q[15];
ry(-3.077599275919977) q[16];
rz(-0.17160149388238463) q[16];
ry(1.562652085699828) q[17];
rz(-1.5710979232131304) q[17];
ry(0.1665960430842408) q[18];
rz(-0.7566908680153777) q[18];
ry(1.4283732354598389) q[19];
rz(-1.3347341579863814) q[19];
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
ry(-1.4464434526612457) q[0];
rz(0.411439946508438) q[0];
ry(-0.3224201416740815) q[1];
rz(1.6281053532136807) q[1];
ry(1.569545741185994) q[2];
rz(0.9778861461475621) q[2];
ry(-0.0013139602821352625) q[3];
rz(-2.187170259407658) q[3];
ry(-0.0005113746820503802) q[4];
rz(1.660144147855967) q[4];
ry(0.02311627014420026) q[5];
rz(2.1954720077005976) q[5];
ry(-0.21682548385675693) q[6];
rz(-0.013255617060760231) q[6];
ry(0.0027883602828691068) q[7];
rz(-2.6830291620004108) q[7];
ry(-2.127285234933873) q[8];
rz(-2.472804346457314) q[8];
ry(1.582817391717649) q[9];
rz(3.1375654936339124) q[9];
ry(-0.9515925013681192) q[10];
rz(-0.08874372679999082) q[10];
ry(-0.038170426562209904) q[11];
rz(-0.599645690880455) q[11];
ry(0.001304977683159514) q[12];
rz(0.2704692706069688) q[12];
ry(0.0001070102914537685) q[13];
rz(2.2293417913319216) q[13];
ry(-3.1412866857379154) q[14];
rz(-2.5689930037561775) q[14];
ry(2.262342084092171) q[15];
rz(-1.5921963819871556) q[15];
ry(0.0005717498646976979) q[16];
rz(0.017344916494897163) q[16];
ry(1.5687757993400824) q[17];
rz(-1.672628240498123) q[17];
ry(-2.2745623521521345) q[18];
rz(-1.6935789950282198) q[18];
ry(-0.0015449905289025684) q[19];
rz(-2.927248740432053) q[19];
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
ry(-1.571166935067163) q[0];
rz(3.114037578431437) q[0];
ry(1.870902814345598) q[1];
rz(-1.0461361179038757) q[1];
ry(-3.1396546710036497) q[2];
rz(0.8441719792958555) q[2];
ry(-0.09584859648688406) q[3];
rz(1.313334198150689) q[3];
ry(0.6552314868329739) q[4];
rz(-1.6148648845191076) q[4];
ry(1.7658499757679644) q[5];
rz(3.092694427432244) q[5];
ry(2.673155913686283) q[6];
rz(0.6168370747553098) q[6];
ry(-0.47488484727716607) q[7];
rz(2.062941601407124) q[7];
ry(3.1388674256195745) q[8];
rz(-0.7567117095164465) q[8];
ry(0.0772107619060467) q[9];
rz(-3.128577708927808) q[9];
ry(1.8246652003935944) q[10];
rz(1.6902372626624915) q[10];
ry(-0.02495683800973758) q[11];
rz(-2.968483475867512) q[11];
ry(1.7268027120211746) q[12];
rz(-1.7805237825556866) q[12];
ry(-1.5469928235123067) q[13];
rz(-0.5363824585069594) q[13];
ry(-1.5819214643616173) q[14];
rz(2.858145640008526) q[14];
ry(2.5356643766603604) q[15];
rz(2.575498266680775) q[15];
ry(3.1393002731994137) q[16];
rz(-0.9371899631054933) q[16];
ry(-1.699771387967) q[17];
rz(-3.1209637157920356) q[17];
ry(-1.5413517376888746) q[18];
rz(1.9366577685238768) q[18];
ry(1.580519547743691) q[19];
rz(-0.20592100594955248) q[19];
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
ry(0.9802435297245307) q[0];
rz(1.591996618947892) q[0];
ry(1.5670458629667432) q[1];
rz(-1.2360667033334665) q[1];
ry(-0.03848602417257466) q[2];
rz(2.7734190338241764) q[2];
ry(-1.5745972487255806) q[3];
rz(2.110540199186338) q[3];
ry(-0.00023466350983181172) q[4];
rz(-1.6705065535917987) q[4];
ry(-0.005054089681186369) q[5];
rz(2.744673981450748) q[5];
ry(-0.25095347340690166) q[6];
rz(-0.2926777296861848) q[6];
ry(-0.032746892260892224) q[7];
rz(-1.463029018636945) q[7];
ry(-1.5975442390188253) q[8];
rz(1.4052016017105329) q[8];
ry(1.5703532877237953) q[9];
rz(1.879115665149091) q[9];
ry(1.542647230998786) q[10];
rz(1.7577626713197554) q[10];
ry(-1.6947596525478141) q[11];
rz(3.077963484992292) q[11];
ry(-2.963200374080087) q[12];
rz(2.860879197651085) q[12];
ry(3.13797472503012) q[13];
rz(2.5591106978960276) q[13];
ry(-0.00027379295231444445) q[14];
rz(1.8396608827684755) q[14];
ry(-3.1310306264098164) q[15];
rz(-2.096789989815996) q[15];
ry(2.6913610152607736) q[16];
rz(0.31453827489981473) q[16];
ry(1.7239805822153373) q[17];
rz(2.1431816481076162) q[17];
ry(-1.5729236582137682) q[18];
rz(-3.1401383172077226) q[18];
ry(-0.012733872607295638) q[19];
rz(0.3425107018072717) q[19];
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
ry(1.5740670217893706) q[0];
rz(-0.22573813271993617) q[0];
ry(-1.2439365498605748) q[1];
rz(-0.7689885359164341) q[1];
ry(3.0064459743527454) q[2];
rz(1.0922218096882583) q[2];
ry(-3.1415559051560087) q[3];
rz(0.051948442719951686) q[3];
ry(0.8639542966221141) q[4];
rz(-1.6714646922676126) q[4];
ry(1.6202531633181732) q[5];
rz(0.4079424519944653) q[5];
ry(-2.7074216871782255) q[6];
rz(0.08200891397812882) q[6];
ry(1.5381140555256474) q[7];
rz(-2.407594462600564) q[7];
ry(1.4079669121693128) q[8];
rz(-0.08678369935311403) q[8];
ry(3.122865915682311) q[9];
rz(2.4350808888185385) q[9];
ry(-0.08685628215445115) q[10];
rz(0.224321150940189) q[10];
ry(-3.1265809421880246) q[11];
rz(3.076949896838165) q[11];
ry(-1.8523779882888392) q[12];
rz(2.03274436824026) q[12];
ry(2.902354468709311) q[13];
rz(2.6880848164049644) q[13];
ry(-2.279900960599856) q[14];
rz(-0.7909439774091456) q[14];
ry(1.5665129716716581) q[15];
rz(-1.6577111966692504) q[15];
ry(-0.2537351171983166) q[16];
rz(1.9572930315298251) q[16];
ry(-2.7938563172887476) q[17];
rz(0.7235255362635643) q[17];
ry(0.1512616491244536) q[18];
rz(-0.00019816086646873288) q[18];
ry(3.1187812623641715) q[19];
rz(1.6138534797434696) q[19];
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
ry(3.125714142969214) q[0];
rz(-2.884799187167944) q[0];
ry(1.5475525417166915) q[1];
rz(-1.1158353862531616) q[1];
ry(0.04114719745713202) q[2];
rz(-1.469937245616079) q[2];
ry(-3.101940229381537) q[3];
rz(2.462669476959506) q[3];
ry(3.130203570364397) q[4];
rz(2.925697260178543) q[4];
ry(-3.140858953236506) q[5];
rz(0.9000102005513284) q[5];
ry(-0.0016161140004244956) q[6];
rz(-1.4400577728716029) q[6];
ry(3.1333561038238384) q[7];
rz(2.359519317519467) q[7];
ry(-2.9776385899150255) q[8];
rz(-0.0663417608698889) q[8];
ry(1.60176445077976) q[9];
rz(-0.6224009220165715) q[9];
ry(0.04715716997496866) q[10];
rz(-0.1784726754121212) q[10];
ry(1.6910436425295847) q[11];
rz(1.3034531481060627) q[11];
ry(-3.0297860361536624) q[12];
rz(2.194780107822539) q[12];
ry(-3.1414741233878014) q[13];
rz(0.8006258524013194) q[13];
ry(0.0012419420977627382) q[14];
rz(0.8771432170979799) q[14];
ry(3.1399514750370665) q[15];
rz(-0.2741328720071927) q[15];
ry(-0.4629007672460426) q[16];
rz(2.0423053794993713) q[16];
ry(1.5726213187273717) q[17];
rz(3.135301857221846) q[17];
ry(1.573802433056757) q[18];
rz(-2.778288950151774) q[18];
ry(-0.22404090811154162) q[19];
rz(1.6434656065070108) q[19];
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
ry(0.04326061221921565) q[0];
rz(0.17302743945094434) q[0];
ry(-0.8310658248064944) q[1];
rz(2.4948994466798227) q[1];
ry(-0.4946929030734344) q[2];
rz(2.962223786843834) q[2];
ry(-0.6891113150940695) q[3];
rz(0.37282830100223124) q[3];
ry(-1.6245218239301646) q[4];
rz(-0.5668221104992615) q[4];
ry(-3.0898823023639137) q[5];
rz(2.487657695812572) q[5];
ry(1.7202974502333248) q[6];
rz(2.0902844818091144) q[6];
ry(-3.084552803374476) q[7];
rz(-2.305512664400469) q[7];
ry(-0.4772704755967796) q[8];
rz(0.010224810395930952) q[8];
ry(-3.1402519210237583) q[9];
rz(2.5668409080453536) q[9];
ry(-1.6203771434116945) q[10];
rz(-1.5695100858522737) q[10];
ry(-0.00894805766023321) q[11];
rz(1.8197170411101729) q[11];
ry(-0.2826279930691488) q[12];
rz(-2.1498898721636843) q[12];
ry(-2.1988485292724755) q[13];
rz(0.04680820326551115) q[13];
ry(-2.6260295718151005) q[14];
rz(-1.9733253271429962) q[14];
ry(-1.3907875920657273) q[15];
rz(1.5147070616305802) q[15];
ry(-0.015989325939743892) q[16];
rz(0.5933518585891615) q[16];
ry(1.2833261463683696) q[17];
rz(-3.140592003038309) q[17];
ry(-0.028608877233006057) q[18];
rz(1.4365540914686) q[18];
ry(-1.7626569718234046) q[19];
rz(1.5732556190510545) q[19];
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
ry(-1.5073048040472217) q[0];
rz(-3.0872431929962008) q[0];
ry(-2.4046162161281996) q[1];
rz(-1.2592579126427719) q[1];
ry(-0.03320954820111233) q[2];
rz(0.01640921534207923) q[2];
ry(-0.1322241636880941) q[3];
rz(1.9127079943603595) q[3];
ry(-0.001827951855005949) q[4];
rz(-2.0180428490572924) q[4];
ry(-3.0817905087500983) q[5];
rz(-2.8192265765005002) q[5];
ry(-3.1388468635410494) q[6];
rz(2.5061081053964744) q[6];
ry(0.0011557993215021156) q[7];
rz(-0.832555187788661) q[7];
ry(-2.9102094273621795) q[8];
rz(1.5642290657474502) q[8];
ry(-1.6926606087384204) q[9];
rz(-2.5347186031294324) q[9];
ry(-1.5450276173745634) q[10];
rz(3.0943351836919764) q[10];
ry(-2.9950169759847522) q[11];
rz(-3.047292070305615) q[11];
ry(0.007444842767708516) q[12];
rz(-2.477378509795798) q[12];
ry(1.5806590203162472) q[13];
rz(-3.140569521130284) q[13];
ry(0.010597442132242561) q[14];
rz(0.4809120270519092) q[14];
ry(3.1199933564817135) q[15];
rz(-2.6880516285003506) q[15];
ry(-3.0116986689876892) q[16];
rz(-0.029914416416158218) q[16];
ry(-1.5720092772909888) q[17];
rz(1.0572109859846002) q[17];
ry(-0.0006613112294182599) q[18];
rz(0.721801315260544) q[18];
ry(-2.2909521168844678) q[19];
rz(-1.5676048679227221) q[19];
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
ry(2.950734290718372) q[0];
rz(-3.0287561593117296) q[0];
ry(-3.1136627965880543) q[1];
rz(-2.8052868703052614) q[1];
ry(1.8787504060177644) q[2];
rz(0.046075073820332164) q[2];
ry(2.3025424750861365) q[3];
rz(0.8864002302511463) q[3];
ry(-3.0761189905798862) q[4];
rz(2.140739154597239) q[4];
ry(-3.0791848184213695) q[5];
rz(-3.113987629900564) q[5];
ry(-1.5201415633199213) q[6];
rz(2.923695961285733) q[6];
ry(0.6066260610439738) q[7];
rz(-0.03699011219290327) q[7];
ry(1.6426336845759684) q[8];
rz(0.006041601546323773) q[8];
ry(-0.0011577046860926643) q[9];
rz(2.2609119414675334) q[9];
ry(-1.5568471591407569) q[10];
rz(-0.004114032834899891) q[10];
ry(3.010922211240494) q[11];
rz(2.217500449089737) q[11];
ry(3.1127673536145597) q[12];
rz(2.2420240530219235) q[12];
ry(-0.7021493584708027) q[13];
rz(3.1382680661770945) q[13];
ry(1.4489217382604318) q[14];
rz(0.08465310134098045) q[14];
ry(0.08643838282876974) q[15];
rz(0.33881482339415747) q[15];
ry(3.130133537366065) q[16];
rz(-0.6697458882735933) q[16];
ry(-1.2412359488392832) q[17];
rz(-1.5896711411616102) q[17];
ry(-0.20976718724266338) q[18];
rz(3.0479447585488915) q[18];
ry(1.5593470368877618) q[19];
rz(3.1349998606316642) q[19];
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
ry(-0.32141712058175964) q[0];
rz(0.3088025868698079) q[0];
ry(-1.553086581553762) q[1];
rz(-0.24111022353838182) q[1];
ry(3.120192662672774) q[2];
rz(2.1708796999093867) q[2];
ry(3.1021098515487755) q[3];
rz(-0.20555354893306751) q[3];
ry(-3.1054956239049183) q[4];
rz(3.0580328885151222) q[4];
ry(3.012597720241273) q[5];
rz(-2.6176995504482545) q[5];
ry(3.1410245582140184) q[6];
rz(0.7804767887636617) q[6];
ry(-0.002892855264185812) q[7];
rz(2.9670758322736384) q[7];
ry(-1.4955346758566208) q[8];
rz(-1.5210475219459019) q[8];
ry(-3.136339486360866) q[9];
rz(-1.300194486073396) q[9];
ry(2.766837414251635) q[10];
rz(1.5250401459916842) q[10];
ry(-0.0011159819686774597) q[11];
rz(0.3612754751799878) q[11];
ry(0.002598592231304992) q[12];
rz(-2.7513184430618414) q[12];
ry(-1.5812669636725454) q[13];
rz(1.5743461766221838) q[13];
ry(-0.007644932201390655) q[14];
rz(-2.4782602409502332) q[14];
ry(2.9996040653291707) q[15];
rz(-3.092338097151216) q[15];
ry(3.140073066797117) q[16];
rz(-0.1734854721627703) q[16];
ry(-0.39331097303941664) q[17];
rz(-2.326859257163578) q[17];
ry(0.05132497901315158) q[18];
rz(-0.6774436853888978) q[18];
ry(1.5764994319110353) q[19];
rz(1.534389944465153) q[19];
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
ry(1.6066876994716326) q[0];
rz(0.1571338491884657) q[0];
ry(-1.5773889602707813) q[1];
rz(0.13938005922880947) q[1];
ry(-0.07230533082094738) q[2];
rz(2.7500441723194204) q[2];
ry(-1.5273265014881587) q[3];
rz(0.14561625408299275) q[3];
ry(1.5548403535751238) q[4];
rz(-2.9878499069465865) q[4];
ry(-1.6530491637442681) q[5];
rz(-2.962486373581797) q[5];
ry(-1.595401415073892) q[6];
rz(-2.985516286741754) q[6];
ry(-0.14278251970466638) q[7];
rz(2.010206103622214) q[7];
ry(-2.4846240002121025) q[8];
rz(1.8217670471496716) q[8];
ry(-1.6068195055153032) q[9];
rz(-2.979509281458711) q[9];
ry(-2.607621477986552) q[10];
rz(1.6853885055845552) q[10];
ry(1.6114761324312257) q[11];
rz(0.1593792642553753) q[11];
ry(1.5974175212045232) q[12];
rz(0.1584378449094682) q[12];
ry(2.715745622871253) q[13];
rz(-1.3486622472949277) q[13];
ry(1.537733838445507) q[14];
rz(0.11536883269074762) q[14];
ry(1.6040748288246363) q[15];
rz(0.16476264963777448) q[15];
ry(1.605361982373589) q[16];
rz(0.1639943029141079) q[16];
ry(-1.5403683586072168) q[17];
rz(0.1477841965036162) q[17];
ry(-1.597809426715581) q[18];
rz(-2.9867318112190713) q[18];
ry(-2.1748253121576173) q[19];
rz(1.7031796657427425) q[19];