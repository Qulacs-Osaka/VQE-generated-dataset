OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
h q[0];
h q[1];
cx q[0],q[1];
rz(-1.3270786320691796) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(1.1794504123254166) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.022654200675594954) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.9607655439167655) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.43989602686287205) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.4242224897009484) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(1.853427847414914) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.9330800077446807) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.10238053128658081) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(-0.003929010627701938) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-0.00718536947077851) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(-1.732665969371474) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(0.9886169276494089) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(-0.15312187593904464) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(1.7443667693531004) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(-0.23106158515575978) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(1.1138768109333081) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(-1.2063582105092856) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(0.15785832676486683) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(0.07914563625020422) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(0.06075784744698665) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(0.534967541907502) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(0.5275307157750467) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(1.5202419040873034) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(-2.5639632969197255) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(-2.672276799581488) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(-1.8814289978721899) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(1.243185993225654) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(0.9948524232638459) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(-0.3766761434497419) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(-0.9340617752406557) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(0.7071940743188286) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(-0.45596644639236217) q[11];
cx q[10],q[11];
h q[0];
h q[2];
cx q[0],q[2];
rz(1.1039854760787902) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(0.0291685088900322) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-1.550419857941362) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.09293899866697661) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(0.9397992842123034) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(-0.2986505254210756) q[3];
cx q[1],q[3];
h q[2];
h q[4];
cx q[2],q[4];
rz(0.00738009514826057) q[4];
cx q[2],q[4];
h q[2];
h q[4];
sdg q[2];
h q[2];
sdg q[4];
h q[4];
cx q[2],q[4];
rz(0.0012256721959642105) q[4];
cx q[2],q[4];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[4];
rz(-0.0028618478308862767) q[4];
cx q[2],q[4];
h q[3];
h q[5];
cx q[3],q[5];
rz(-0.0006945940301888512) q[5];
cx q[3],q[5];
h q[3];
h q[5];
sdg q[3];
h q[3];
sdg q[5];
h q[5];
cx q[3],q[5];
rz(-0.0016980368404346828) q[5];
cx q[3],q[5];
h q[3];
s q[3];
h q[5];
s q[5];
cx q[3],q[5];
rz(0.0009894851568676493) q[5];
cx q[3],q[5];
h q[4];
h q[6];
cx q[4],q[6];
rz(-2.019536875091759) q[6];
cx q[4],q[6];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[6];
rz(0.5342372271822455) q[6];
cx q[4],q[6];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[6];
rz(2.7739126068534414) q[6];
cx q[4],q[6];
h q[5];
h q[7];
cx q[5],q[7];
rz(0.03291520494667847) q[7];
cx q[5],q[7];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[7];
rz(-0.1253770410564421) q[7];
cx q[5],q[7];
h q[5];
s q[5];
h q[7];
s q[7];
cx q[5],q[7];
rz(-2.398335486091459) q[7];
cx q[5],q[7];
h q[6];
h q[8];
cx q[6],q[8];
rz(0.09717705580747223) q[8];
cx q[6],q[8];
h q[6];
h q[8];
sdg q[6];
h q[6];
sdg q[8];
h q[8];
cx q[6],q[8];
rz(-0.006593323747502301) q[8];
cx q[6],q[8];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[8];
rz(-0.8349316640383523) q[8];
cx q[6],q[8];
h q[7];
h q[9];
cx q[7],q[9];
rz(0.00894711776367472) q[9];
cx q[7],q[9];
h q[7];
h q[9];
sdg q[7];
h q[7];
sdg q[9];
h q[9];
cx q[7],q[9];
rz(0.007619255756361932) q[9];
cx q[7],q[9];
h q[7];
s q[7];
h q[9];
s q[9];
cx q[7],q[9];
rz(-0.4460547263681162) q[9];
cx q[7],q[9];
h q[8];
h q[10];
cx q[8],q[10];
rz(0.011523666272004084) q[10];
cx q[8],q[10];
h q[8];
h q[10];
sdg q[8];
h q[8];
sdg q[10];
h q[10];
cx q[8],q[10];
rz(0.2676503384994984) q[10];
cx q[8],q[10];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[10];
rz(0.22929759034613606) q[10];
cx q[8],q[10];
h q[9];
h q[11];
cx q[9],q[11];
rz(0.46256902665693556) q[11];
cx q[9],q[11];
h q[9];
h q[11];
sdg q[9];
h q[9];
sdg q[11];
h q[11];
cx q[9],q[11];
rz(0.5699250250938401) q[11];
cx q[9],q[11];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[9],q[11];
rz(-0.5220793361586633) q[11];
cx q[9],q[11];
rx(0.2476886275860848) q[0];
rz(-0.15319654632447657) q[0];
rx(1.3780322344022629) q[1];
rz(0.894270731949801) q[1];
rx(0.08085689059858402) q[2];
rz(0.7692290000950937) q[2];
rx(0.18091328478466331) q[3];
rz(0.4501820834252933) q[3];
rx(-0.032900688721034535) q[4];
rz(1.6966387281078037) q[4];
rx(0.024183352635431258) q[5];
rz(0.697942664002065) q[5];
rx(0.018732382783497703) q[6];
rz(-0.22725378217135858) q[6];
rx(-1.5414710898903312) q[7];
rz(-1.5471959325830398) q[7];
rx(-2.0244447008612187) q[8];
rz(-5.740601900795934e-05) q[8];
rx(0.001991636290092679) q[9];
rz(-1.197748914901852) q[9];
rx(0.0006603527203986861) q[10];
rz(0.18614434984321646) q[10];
rx(0.0007264973589037427) q[11];
rz(0.19362544535177545) q[11];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.06719692681335676) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(2.158576015331939) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(1.5106031139557268) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.7309969257012261) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.4916672046864434) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.28179775361682524) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.880538866530997) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.6287875781854855) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.725764841684464) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(0.00160786100643937) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-0.0008237232886357884) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(0.012621347719950266) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(1.2729662820087795) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(-1.6168820068570349) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(0.20669135601059613) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(0.005729487968588298) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(-0.00904332364801276) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(-1.1528366708041324) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(-0.0014056999497885014) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(-0.003491019202865519) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(-0.0008251490008599913) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(-1.1155269543682285) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(-0.011327783590330157) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(0.0009918933489426053) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(0.3229909879558213) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(-0.5551244460951922) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(1.3078732084134626) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(-0.09038978749255465) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(0.46511476964819415) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(-0.6216142286049237) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(2.544331581209827) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(0.7217261559902086) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(-3.078201785432557) q[11];
cx q[10],q[11];
h q[0];
h q[2];
cx q[0],q[2];
rz(0.7232412542061233) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.1414146971377387) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-0.45815233600477134) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.37416453461362675) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(0.6717243163983858) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(0.5311633452403406) q[3];
cx q[1],q[3];
h q[2];
h q[4];
cx q[2],q[4];
rz(0.050083349687523954) q[4];
cx q[2],q[4];
h q[2];
h q[4];
sdg q[2];
h q[2];
sdg q[4];
h q[4];
cx q[2],q[4];
rz(-0.1754397889736855) q[4];
cx q[2],q[4];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[4];
rz(-0.17684827400975614) q[4];
cx q[2],q[4];
h q[3];
h q[5];
cx q[3],q[5];
rz(-0.0067011052995415024) q[5];
cx q[3],q[5];
h q[3];
h q[5];
sdg q[3];
h q[3];
sdg q[5];
h q[5];
cx q[3],q[5];
rz(-0.0011974281539781483) q[5];
cx q[3],q[5];
h q[3];
s q[3];
h q[5];
s q[5];
cx q[3],q[5];
rz(-0.05687879903016332) q[5];
cx q[3],q[5];
h q[4];
h q[6];
cx q[4],q[6];
rz(-3.117237914145944) q[6];
cx q[4],q[6];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[6];
rz(0.32459207900407405) q[6];
cx q[4],q[6];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[6];
rz(0.3321368118377641) q[6];
cx q[4],q[6];
h q[5];
h q[7];
cx q[5],q[7];
rz(0.0036396359552490418) q[7];
cx q[5],q[7];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[7];
rz(1.5722446082820358) q[7];
cx q[5],q[7];
h q[5];
s q[5];
h q[7];
s q[7];
cx q[5],q[7];
rz(-0.011548162143805904) q[7];
cx q[5],q[7];
h q[6];
h q[8];
cx q[6],q[8];
rz(-0.010726742920424663) q[8];
cx q[6],q[8];
h q[6];
h q[8];
sdg q[6];
h q[6];
sdg q[8];
h q[8];
cx q[6],q[8];
rz(-0.09258298191374406) q[8];
cx q[6],q[8];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[8];
rz(0.09472379050934489) q[8];
cx q[6],q[8];
h q[7];
h q[9];
cx q[7],q[9];
rz(0.0060035748116105195) q[9];
cx q[7],q[9];
h q[7];
h q[9];
sdg q[7];
h q[7];
sdg q[9];
h q[9];
cx q[7],q[9];
rz(0.07328351738175329) q[9];
cx q[7],q[9];
h q[7];
s q[7];
h q[9];
s q[9];
cx q[7],q[9];
rz(-0.024499147212247174) q[9];
cx q[7],q[9];
h q[8];
h q[10];
cx q[8],q[10];
rz(0.7272068966900315) q[10];
cx q[8],q[10];
h q[8];
h q[10];
sdg q[8];
h q[8];
sdg q[10];
h q[10];
cx q[8],q[10];
rz(0.506322743038522) q[10];
cx q[8],q[10];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[10];
rz(-0.5168285300233184) q[10];
cx q[8],q[10];
h q[9];
h q[11];
cx q[9],q[11];
rz(0.0437903684118855) q[11];
cx q[9],q[11];
h q[9];
h q[11];
sdg q[9];
h q[9];
sdg q[11];
h q[11];
cx q[9],q[11];
rz(0.5206457566052377) q[11];
cx q[9],q[11];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[9],q[11];
rz(-0.5078640159205254) q[11];
cx q[9],q[11];
rx(-0.22825448923151506) q[0];
rz(-0.34709040303997923) q[0];
rx(-0.5591670030396493) q[1];
rz(-0.17008051987516723) q[1];
rx(2.143078275387851) q[2];
rz(0.006751053353556157) q[2];
rx(-0.7753750074212753) q[3];
rz(1.418767178992057) q[3];
rx(-0.9953888761912617) q[4];
rz(-0.00927054280278743) q[4];
rx(0.5512022722716057) q[5];
rz(0.010975251530697432) q[5];
rx(-1.0097492071922705) q[6];
rz(0.02016314669022514) q[6];
rx(-0.012277467251435255) q[7];
rz(-1.806746547432536) q[7];
rx(-0.000604588654198053) q[8];
rz(-2.077289377678185) q[8];
rx(0.004699782952649845) q[9];
rz(-0.023610092699433444) q[9];
rx(0.0010309785866551065) q[10];
rz(-1.0878666106925716) q[10];
rx(-0.000606302418844213) q[11];
rz(-0.21895388235998456) q[11];
h q[0];
h q[1];
cx q[0],q[1];
rz(-1.2031568330382918) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(1.6717479974473493) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.40484219553232503) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(2.3465756134542857) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.13391504247856317) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.13073292928048663) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.7537720445143193) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.002105173254301828) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.0008800981797726366) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(0.1473195562990334) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(0.054651289280511224) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(-0.05531503467411454) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(1.485555545557436) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(-1.6792194060548993) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(1.4783884160732694) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(-0.2224311290508728) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(2.3040522710837585) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(0.8532533074268557) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(0.033917980590179336) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(0.007546459278920124) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(0.026385861498281912) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(0.35425475868314343) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(-0.28755384293959285) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(0.2864012493754435) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(1.629591969304415) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(-1.6144221987519176) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(1.3831282302532015) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(0.4054519486636869) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(-0.009850754915434752) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(0.05713418005520937) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(1.0221232699740257) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(-1.629120322793847) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(0.8063221828795744) q[11];
cx q[10],q[11];
h q[0];
h q[2];
cx q[0],q[2];
rz(0.5638975142835939) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(0.17771420885545872) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-0.1798089977858579) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.905765861345976) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(1.346120142550701) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(-1.344059409310561) q[3];
cx q[1],q[3];
h q[2];
h q[4];
cx q[2],q[4];
rz(0.04686575891968242) q[4];
cx q[2],q[4];
h q[2];
h q[4];
sdg q[2];
h q[2];
sdg q[4];
h q[4];
cx q[2],q[4];
rz(-0.04980188055045151) q[4];
cx q[2],q[4];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[4];
rz(0.04760028432619643) q[4];
cx q[2],q[4];
h q[3];
h q[5];
cx q[3],q[5];
rz(-0.027156803679868356) q[5];
cx q[3],q[5];
h q[3];
h q[5];
sdg q[3];
h q[3];
sdg q[5];
h q[5];
cx q[3],q[5];
rz(-0.13442227812877652) q[5];
cx q[3],q[5];
h q[3];
s q[3];
h q[5];
s q[5];
cx q[3],q[5];
rz(0.13416106458532298) q[5];
cx q[3],q[5];
h q[4];
h q[6];
cx q[4],q[6];
rz(-0.11386665327432643) q[6];
cx q[4],q[6];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[6];
rz(0.22254694368480352) q[6];
cx q[4],q[6];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[6];
rz(0.21382985616748207) q[6];
cx q[4],q[6];
h q[5];
h q[7];
cx q[5],q[7];
rz(-0.0030057133719643874) q[7];
cx q[5],q[7];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[7];
rz(0.0015012072268777606) q[7];
cx q[5],q[7];
h q[5];
s q[5];
h q[7];
s q[7];
cx q[5],q[7];
rz(0.0014482666163264462) q[7];
cx q[5],q[7];
h q[6];
h q[8];
cx q[6],q[8];
rz(-0.0706682223094617) q[8];
cx q[6],q[8];
h q[6];
h q[8];
sdg q[6];
h q[6];
sdg q[8];
h q[8];
cx q[6],q[8];
rz(-0.00247831998899677) q[8];
cx q[6],q[8];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[8];
rz(0.0006814891618611625) q[8];
cx q[6],q[8];
h q[7];
h q[9];
cx q[7],q[9];
rz(0.4638467335241372) q[9];
cx q[7],q[9];
h q[7];
h q[9];
sdg q[7];
h q[7];
sdg q[9];
h q[9];
cx q[7],q[9];
rz(0.339320698526772) q[9];
cx q[7],q[9];
h q[7];
s q[7];
h q[9];
s q[9];
cx q[7],q[9];
rz(0.3381942114123562) q[9];
cx q[7],q[9];
h q[8];
h q[10];
cx q[8],q[10];
rz(-1.4116340999448544) q[10];
cx q[8],q[10];
h q[8];
h q[10];
sdg q[8];
h q[8];
sdg q[10];
h q[10];
cx q[8],q[10];
rz(0.15685985885037662) q[10];
cx q[8],q[10];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[10];
rz(-0.8814783409039889) q[10];
cx q[8],q[10];
h q[9];
h q[11];
cx q[9],q[11];
rz(-1.7215854603406746) q[11];
cx q[9],q[11];
h q[9];
h q[11];
sdg q[9];
h q[9];
sdg q[11];
h q[11];
cx q[9],q[11];
rz(-0.771601985634072) q[11];
cx q[9],q[11];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[9],q[11];
rz(0.6029957430715653) q[11];
cx q[9],q[11];
rx(-1.4403047260757187) q[0];
rz(0.042194474583057716) q[0];
rx(-1.2955517097673672) q[1];
rz(0.013965557645223486) q[1];
rx(-2.1328566681613004) q[2];
rz(-0.06398691203268146) q[2];
rx(-1.1110267116766908) q[3];
rz(0.05773670134657973) q[3];
rx(2.1055555086264137) q[4];
rz(0.10838046643130948) q[4];
rx(0.9946468995921125) q[5];
rz(0.0711219822283531) q[5];
rx(2.147524791212179) q[6];
rz(-0.1169967879898147) q[6];
rx(-0.0052019865973001915) q[7];
rz(0.34276129602031163) q[7];
rx(0.0030277413345873847) q[8];
rz(-0.017117434471491663) q[8];
rx(-0.0006824061991327992) q[9];
rz(0.3036903884997267) q[9];
rx(0.0005672492846904832) q[10];
rz(-1.6070452497485233) q[10];
rx(0.0014407689721381033) q[11];
rz(-1.9887716429555005) q[11];
h q[0];
h q[1];
cx q[0],q[1];
rz(1.7579191259188511) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(1.5836701430951656) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(1.554896043577164) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.011445719682396317) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.0919733907452745) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.0815744528701624) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.23993006151059007) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.24113208788748092) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.25549362608352033) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(0.16336918898199923) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(0.18381931895937503) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(0.18792214992231662) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(1.6523261011470716) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(-1.5119550588135562) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(1.523141450330121) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(-0.009347138326015789) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(-0.05466365481468129) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(0.06069753127334003) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(-0.0486088864468247) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(0.035194915024333785) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(0.037601922729452494) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(0.07080743622933662) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(0.02028263539729575) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(0.02713694084952535) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(0.9510208308798412) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(0.002104286693246976) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(-0.04017204063706973) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(0.9225462189358125) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(0.07640975959603583) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(0.16322444874361297) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(0.18647285269595665) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(0.04137183761882334) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(0.10044092238692974) q[11];
cx q[10],q[11];
h q[0];
h q[2];
cx q[0],q[2];
rz(0.25030144373212926) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(3.0787346199738757) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-0.08010740762352037) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(0.3107417881954258) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(0.3970165126948052) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(0.39003291438050197) q[3];
cx q[1],q[3];
h q[2];
h q[4];
cx q[2],q[4];
rz(0.2617197537978541) q[4];
cx q[2],q[4];
h q[2];
h q[4];
sdg q[2];
h q[2];
sdg q[4];
h q[4];
cx q[2],q[4];
rz(-0.2953746285023014) q[4];
cx q[2],q[4];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[4];
rz(-0.2957257645613073) q[4];
cx q[2],q[4];
h q[3];
h q[5];
cx q[3],q[5];
rz(-0.18377392194665548) q[5];
cx q[3],q[5];
h q[3];
h q[5];
sdg q[3];
h q[3];
sdg q[5];
h q[5];
cx q[3],q[5];
rz(0.15307006562917203) q[5];
cx q[3],q[5];
h q[3];
s q[3];
h q[5];
s q[5];
cx q[3],q[5];
rz(-0.14816057990195158) q[5];
cx q[3],q[5];
h q[4];
h q[6];
cx q[4],q[6];
rz(-2.5998366632749628) q[6];
cx q[4],q[6];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[6];
rz(0.5733296461437541) q[6];
cx q[4],q[6];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[6];
rz(0.5762816841494315) q[6];
cx q[4],q[6];
h q[5];
h q[7];
cx q[5],q[7];
rz(-0.6510513522591379) q[7];
cx q[5],q[7];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[7];
rz(0.6749633960249757) q[7];
cx q[5],q[7];
h q[5];
s q[5];
h q[7];
s q[7];
cx q[5],q[7];
rz(-0.6978261670869069) q[7];
cx q[5],q[7];
h q[6];
h q[8];
cx q[6],q[8];
rz(-0.35432789906948586) q[8];
cx q[6],q[8];
h q[6];
h q[8];
sdg q[6];
h q[6];
sdg q[8];
h q[8];
cx q[6],q[8];
rz(0.3685421972458476) q[8];
cx q[6],q[8];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[8];
rz(0.37127356212185325) q[8];
cx q[6],q[8];
h q[7];
h q[9];
cx q[7],q[9];
rz(0.4340875758331921) q[9];
cx q[7],q[9];
h q[7];
h q[9];
sdg q[7];
h q[7];
sdg q[9];
h q[9];
cx q[7],q[9];
rz(0.43434069415262594) q[9];
cx q[7],q[9];
h q[7];
s q[7];
h q[9];
s q[9];
cx q[7],q[9];
rz(0.43082504316591075) q[9];
cx q[7],q[9];
h q[8];
h q[10];
cx q[8],q[10];
rz(2.2770729569959265) q[10];
cx q[8],q[10];
h q[8];
h q[10];
sdg q[8];
h q[8];
sdg q[10];
h q[10];
cx q[8],q[10];
rz(0.9563518715531291) q[10];
cx q[8],q[10];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[10];
rz(0.9676905268617414) q[10];
cx q[8],q[10];
h q[9];
h q[11];
cx q[9],q[11];
rz(0.8003182422099633) q[11];
cx q[9],q[11];
h q[9];
h q[11];
sdg q[9];
h q[9];
sdg q[11];
h q[11];
cx q[9],q[11];
rz(0.8016759451189442) q[11];
cx q[9],q[11];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[9],q[11];
rz(0.8084191954761474) q[11];
cx q[9],q[11];
rx(-2.2216867360710126) q[0];
rz(-0.2525411695307533) q[0];
rx(0.11134335138517058) q[1];
rz(-0.3480646031791767) q[1];
rx(0.10954506606811308) q[2];
rz(-0.29718209810696244) q[2];
rx(0.10300761244827793) q[3];
rz(-0.3366265947634848) q[3];
rx(0.08407210194157509) q[4];
rz(-0.28009470612190557) q[4];
rx(3.0412296744941516) q[5];
rz(2.8042777653371243) q[5];
rx(0.09071702512456094) q[6];
rz(-0.3056596772734679) q[6];
rx(0.10470620062726907) q[7];
rz(-0.3419757566225181) q[7];
rx(0.09200919480051482) q[8];
rz(-0.31392757503537333) q[8];
rx(0.10638048181254545) q[9];
rz(-0.34347299911995044) q[9];
rx(-3.0522556450968112) q[10];
rz(-0.30033318661358344) q[10];
rx(0.10683275396393094) q[11];
rz(-0.3493522958184467) q[11];