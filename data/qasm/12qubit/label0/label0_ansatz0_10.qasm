OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
cx q[0],q[1];
rz(-0.012037622196888871) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.06873140212026878) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.03920896785798688) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.0886075105638116) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.0984565085071767) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.09327019375117006) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.015562964199654885) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.028608507947744836) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.03321228003918103) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.040615737556459225) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.028519910188610005) q[11];
cx q[10],q[11];
h q[0];
rz(0.928572309845371) q[0];
h q[0];
h q[1];
rz(0.7138679503232928) q[1];
h q[1];
h q[2];
rz(0.9492270571028212) q[2];
h q[2];
h q[3];
rz(0.7167873491118665) q[3];
h q[3];
h q[4];
rz(-0.038534068467753334) q[4];
h q[4];
h q[5];
rz(1.2771458004414826) q[5];
h q[5];
h q[6];
rz(0.9622829960072631) q[6];
h q[6];
h q[7];
rz(0.5974255972719786) q[7];
h q[7];
h q[8];
rz(0.5268632902795719) q[8];
h q[8];
h q[9];
rz(-0.021448023357121267) q[9];
h q[9];
h q[10];
rz(0.07300970138437149) q[10];
h q[10];
h q[11];
rz(0.11415540011105038) q[11];
h q[11];
rz(-0.1461306223479741) q[0];
rz(-0.25512098795338234) q[1];
rz(-0.072787339249814) q[2];
rz(0.01923707906038757) q[3];
rz(-0.16208253896319508) q[4];
rz(-0.08213170759542587) q[5];
rz(-0.5783039129500207) q[6];
rz(-0.15043126916468458) q[7];
rz(-0.19615397316821354) q[8];
rz(-0.01496229652983418) q[9];
rz(-0.7472950689284839) q[10];
rz(-0.32594447801456866) q[11];
cx q[0],q[1];
rz(-0.026118247030498545) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.03163637708099805) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(0.05041549763181597) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.2091434061838366) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.059468391272052654) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.5427298777925896) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(0.003277465702935211) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(0.049815901798022706) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.1678535144721196) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.2429428649698187) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.23198599632484374) q[11];
cx q[10],q[11];
h q[0];
rz(0.6444168452009545) q[0];
h q[0];
h q[1];
rz(0.6233528532212952) q[1];
h q[1];
h q[2];
rz(0.654491618587791) q[2];
h q[2];
h q[3];
rz(0.3990841452617756) q[3];
h q[3];
h q[4];
rz(0.6526823365640311) q[4];
h q[4];
h q[5];
rz(0.8557456973607027) q[5];
h q[5];
h q[6];
rz(0.6907796222236317) q[6];
h q[6];
h q[7];
rz(0.582165464894731) q[7];
h q[7];
h q[8];
rz(0.40872979746368954) q[8];
h q[8];
h q[9];
rz(0.6525409338762623) q[9];
h q[9];
h q[10];
rz(0.5950713992949421) q[10];
h q[10];
h q[11];
rz(0.4447387759005966) q[11];
h q[11];
rz(-0.19629185783133168) q[0];
rz(-0.10909644763531771) q[1];
rz(-0.25084471388151874) q[2];
rz(-0.023575076696464015) q[3];
rz(-0.05887704749374005) q[4];
rz(-0.00881041165161652) q[5];
rz(-0.7876835099423929) q[6];
rz(-0.3457669026637061) q[7];
rz(-0.3440567625547436) q[8];
rz(-0.24288700011391615) q[9];
rz(-0.9462600681150808) q[10];
rz(-0.5161005820329632) q[11];
cx q[0],q[1];
rz(0.18055572635728748) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(0.012506175408957066) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.04153983646886671) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.3327925898767854) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(0.21506317237335815) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.47230244002717914) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.006663308649752303) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(0.20468026859435856) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(0.0004618133552956399) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(0.34055014360955765) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(0.3542680221185058) q[11];
cx q[10],q[11];
h q[0];
rz(0.42373563787454877) q[0];
h q[0];
h q[1];
rz(0.9342731051240785) q[1];
h q[1];
h q[2];
rz(0.13932223095895208) q[2];
h q[2];
h q[3];
rz(0.24883374214796106) q[3];
h q[3];
h q[4];
rz(0.27500207467890075) q[4];
h q[4];
h q[5];
rz(0.8970980635741376) q[5];
h q[5];
h q[6];
rz(0.34415998689474053) q[6];
h q[6];
h q[7];
rz(0.2566708454697362) q[7];
h q[7];
h q[8];
rz(0.33451775470286316) q[8];
h q[8];
h q[9];
rz(0.6718720173056236) q[9];
h q[9];
h q[10];
rz(0.47840357141730583) q[10];
h q[10];
h q[11];
rz(0.3132763368112732) q[11];
h q[11];
rz(-0.4056707239383918) q[0];
rz(-0.03965673764328331) q[1];
rz(-0.3986172791983929) q[2];
rz(-0.16984295501276087) q[3];
rz(-0.08923667872050706) q[4];
rz(0.00982085855345957) q[5];
rz(-0.5927842873443596) q[6];
rz(-0.4328288691328436) q[7];
rz(-0.42499079752811814) q[8];
rz(-0.6491985798401273) q[9];
rz(-0.9841319218517347) q[10];
rz(-0.6027006982027936) q[11];
cx q[0],q[1];
rz(-0.19083506463735317) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(0.22070494255202955) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.2285796659832906) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.5308359193568609) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.08297866654463641) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(0.022060229419900785) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(0.003908934392097729) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(0.3414141441516519) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(0.006537859639734629) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(0.0664416990976293) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(0.636354657113444) q[11];
cx q[10],q[11];
h q[0];
rz(0.13620391537433071) q[0];
h q[0];
h q[1];
rz(0.9406465974059945) q[1];
h q[1];
h q[2];
rz(-0.07360915640439183) q[2];
h q[2];
h q[3];
rz(0.14444840067031572) q[3];
h q[3];
h q[4];
rz(-0.33492688242107826) q[4];
h q[4];
h q[5];
rz(0.979448340334771) q[5];
h q[5];
h q[6];
rz(0.3803541224032644) q[6];
h q[6];
h q[7];
rz(0.2990647748583162) q[7];
h q[7];
h q[8];
rz(0.3794715601400905) q[8];
h q[8];
h q[9];
rz(0.41391901788689933) q[9];
h q[9];
h q[10];
rz(0.47260420208297793) q[10];
h q[10];
h q[11];
rz(0.3821048628223334) q[11];
h q[11];
rz(-0.499629809276518) q[0];
rz(0.1528725398442198) q[1];
rz(-0.2959091621672858) q[2];
rz(-0.2373021191933868) q[3];
rz(-0.12400261008441249) q[4];
rz(-0.25340696880341296) q[5];
rz(-0.12182124587162464) q[6];
rz(-0.5595531407905344) q[7];
rz(-0.42249673333503956) q[8];
rz(-0.6514697313567795) q[9];
rz(-0.4516933101995144) q[10];
rz(-0.5502840791074672) q[11];
cx q[0],q[1];
rz(-0.1918319685761208) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(0.11142642977799247) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.13030022603225147) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.706160419426662) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.30597006215686956) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.27039066543562923) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.29352956719654) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.2606832400697502) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(0.12809467479445513) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(0.026957598520515554) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(0.15507756049937396) q[11];
cx q[10],q[11];
h q[0];
rz(-0.07718461968833032) q[0];
h q[0];
h q[1];
rz(0.7967786313395853) q[1];
h q[1];
h q[2];
rz(0.07411890082856935) q[2];
h q[2];
h q[3];
rz(-0.09078011000859247) q[3];
h q[3];
h q[4];
rz(-0.07824731804042845) q[4];
h q[4];
h q[5];
rz(0.5403318506602346) q[5];
h q[5];
h q[6];
rz(0.27929242061020554) q[6];
h q[6];
h q[7];
rz(-0.27671976879204274) q[7];
h q[7];
h q[8];
rz(0.07816246737976365) q[8];
h q[8];
h q[9];
rz(-0.08602867491287) q[9];
h q[9];
h q[10];
rz(0.6777300604319396) q[10];
h q[10];
h q[11];
rz(0.3968922226480486) q[11];
h q[11];
rz(-0.44527260884414027) q[0];
rz(0.27470958075606594) q[1];
rz(-0.3363670970394185) q[2];
rz(0.43912780431557863) q[3];
rz(-0.2833916045461263) q[4];
rz(-0.17366475494598316) q[5];
rz(0.06633702590120087) q[6];
rz(-0.21415472191400775) q[7];
rz(-0.321865830205228) q[8];
rz(-0.39291234727169627) q[9];
rz(-0.13700902865518444) q[10];
rz(-0.37251051668435997) q[11];
cx q[0],q[1];
rz(-0.12071179067356518) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.045676464772394766) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(0.3517651111214163) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.3093139624525263) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.6134203984403122) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.24302009550686687) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.03306005224387926) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(0.03118356641872261) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(0.11964052734482267) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.2748344800692469) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.7144178828769145) q[11];
cx q[10],q[11];
h q[0];
rz(-0.2285110744814592) q[0];
h q[0];
h q[1];
rz(0.5904830210378972) q[1];
h q[1];
h q[2];
rz(-0.023742668878088936) q[2];
h q[2];
h q[3];
rz(0.495758905604957) q[3];
h q[3];
h q[4];
rz(-0.7116081125527468) q[4];
h q[4];
h q[5];
rz(-0.04509479331152321) q[5];
h q[5];
h q[6];
rz(0.04218323141099739) q[6];
h q[6];
h q[7];
rz(0.3164814077653715) q[7];
h q[7];
h q[8];
rz(0.04440169248944815) q[8];
h q[8];
h q[9];
rz(-0.6494943597538834) q[9];
h q[9];
h q[10];
rz(0.22675737741562244) q[10];
h q[10];
h q[11];
rz(0.1152090042834795) q[11];
h q[11];
rz(-0.362236858129318) q[0];
rz(0.19941740122925178) q[1];
rz(-0.06534551720962269) q[2];
rz(0.5144717269330423) q[3];
rz(0.22351153991454129) q[4];
rz(-0.15100110401774108) q[5];
rz(0.04105129929613756) q[6];
rz(-0.21207682326275065) q[7];
rz(-0.1731797470061082) q[8];
rz(0.05808283630267586) q[9];
rz(-0.09013837631944167) q[10];
rz(-0.2127995977467281) q[11];
cx q[0],q[1];
rz(0.12330458386171267) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(0.1397580775819191) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.03234898744894936) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(0.4682579702388219) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.5918097755639496) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(0.07669408696157357) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.04504685007336345) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(0.03984797751544206) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.40294249316950875) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.052132896249635435) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.7132849129965166) q[11];
cx q[10],q[11];
h q[0];
rz(-0.4600131401251973) q[0];
h q[0];
h q[1];
rz(0.4543552066177174) q[1];
h q[1];
h q[2];
rz(-0.2923978079809996) q[2];
h q[2];
h q[3];
rz(-0.07476162124544518) q[3];
h q[3];
h q[4];
rz(-0.7844816753356357) q[4];
h q[4];
h q[5];
rz(0.058227905411652785) q[5];
h q[5];
h q[6];
rz(0.2079736671759629) q[6];
h q[6];
h q[7];
rz(0.6210019419575201) q[7];
h q[7];
h q[8];
rz(-0.47472256774838156) q[8];
h q[8];
h q[9];
rz(-0.3391532230306223) q[9];
h q[9];
h q[10];
rz(0.15308532645477713) q[10];
h q[10];
h q[11];
rz(0.036712291507938234) q[11];
h q[11];
rz(-0.20119336520840633) q[0];
rz(0.012888863680148476) q[1];
rz(0.06267806920185914) q[2];
rz(0.3980742258396064) q[3];
rz(0.38272928041679377) q[4];
rz(0.19481254056851952) q[5];
rz(-0.006468774202197712) q[6];
rz(-0.014215676446984108) q[7];
rz(0.1332737619241183) q[8];
rz(-0.07979596792026533) q[9];
rz(-0.014499339299754556) q[10];
rz(-0.08445918094983232) q[11];
cx q[0],q[1];
rz(0.33742196539415603) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(0.04708046389180821) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(0.05890878745837021) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(0.9092096421063305) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(0.31424683368022177) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.29265572698648057) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.10774667640499246) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(0.6577712993362699) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(0.25710361996733844) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(0.010573928115095468) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.48584718754741857) q[11];
cx q[10],q[11];
h q[0];
rz(-0.5302051435143568) q[0];
h q[0];
h q[1];
rz(0.06953429295594742) q[1];
h q[1];
h q[2];
rz(-0.29644571368420847) q[2];
h q[2];
h q[3];
rz(-0.6575816653914077) q[3];
h q[3];
h q[4];
rz(-0.8137431613813276) q[4];
h q[4];
h q[5];
rz(-0.09989423713432663) q[5];
h q[5];
h q[6];
rz(-0.31689610935068957) q[6];
h q[6];
h q[7];
rz(-0.5093447262189433) q[7];
h q[7];
h q[8];
rz(-0.24838914758537628) q[8];
h q[8];
h q[9];
rz(-0.07127789269428585) q[9];
h q[9];
h q[10];
rz(0.26825143925876355) q[10];
h q[10];
h q[11];
rz(0.10589874589101392) q[11];
h q[11];
rz(-0.024243405310776723) q[0];
rz(-0.06361049209989864) q[1];
rz(0.05065870463397351) q[2];
rz(0.23171805391876682) q[3];
rz(0.10598162473612076) q[4];
rz(0.27271255557109536) q[5];
rz(-0.0037005827475395576) q[6];
rz(0.013850495853097932) q[7];
rz(-0.13202818425039398) q[8];
rz(-0.14698429054261639) q[9];
rz(-0.11895170917230398) q[10];
rz(-0.06800191065955344) q[11];
cx q[0],q[1];
rz(0.36446889331751453) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.14380976652044392) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.030263053237806755) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(0.6055308629015951) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(0.6683451915064832) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(0.08033482734676864) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.11773214231174929) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.6331239322524431) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(0.2795872478671542) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(0.1857436608650459) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.2339272001297022) q[11];
cx q[10],q[11];
h q[0];
rz(-0.5949260734308635) q[0];
h q[0];
h q[1];
rz(0.061528193048469654) q[1];
h q[1];
h q[2];
rz(-0.3849899728249296) q[2];
h q[2];
h q[3];
rz(-0.7834597756522781) q[3];
h q[3];
h q[4];
rz(-0.27181983014688077) q[4];
h q[4];
h q[5];
rz(0.057185902144536886) q[5];
h q[5];
h q[6];
rz(0.06205804162200475) q[6];
h q[6];
h q[7];
rz(-0.07155883655373518) q[7];
h q[7];
h q[8];
rz(-0.6572978918709962) q[8];
h q[8];
h q[9];
rz(0.009560225744944162) q[9];
h q[9];
h q[10];
rz(0.4575503511516673) q[10];
h q[10];
h q[11];
rz(-0.0021693517658034) q[11];
h q[11];
rz(0.21102374588756828) q[0];
rz(-0.06968606883373767) q[1];
rz(-0.07967095557619201) q[2];
rz(0.3431117348608999) q[3];
rz(-0.05216883110687255) q[4];
rz(-0.3378045930027578) q[5];
rz(0.02958839362979319) q[6];
rz(-0.03390426929459355) q[7];
rz(0.3485700416543642) q[8];
rz(0.13975797485980698) q[9];
rz(-0.12668094120622744) q[10];
rz(0.005775744857516013) q[11];
cx q[0],q[1];
rz(0.4484158524618) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.12328766021204673) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(0.20949555296588368) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(1.5570143645967804) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(0.1677034080436447) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(0.3643781374298017) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(0.48184526115215315) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(0.11202517751466515) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.4662771717373966) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(0.07030620198469917) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.02297241469559104) q[11];
cx q[10],q[11];
h q[0];
rz(-0.4773136364193175) q[0];
h q[0];
h q[1];
rz(-0.030493454676099513) q[1];
h q[1];
h q[2];
rz(-1.106222076779969) q[2];
h q[2];
h q[3];
rz(0.26850632553102705) q[3];
h q[3];
h q[4];
rz(-0.5339149242221269) q[4];
h q[4];
h q[5];
rz(-0.07784733734664184) q[5];
h q[5];
h q[6];
rz(-0.033907469348990145) q[6];
h q[6];
h q[7];
rz(-0.6820709192241806) q[7];
h q[7];
h q[8];
rz(-0.19336832261777304) q[8];
h q[8];
h q[9];
rz(-0.45133146791641493) q[9];
h q[9];
h q[10];
rz(0.3635019219668171) q[10];
h q[10];
h q[11];
rz(-0.06621214113673012) q[11];
h q[11];
rz(0.3416913265142189) q[0];
rz(-0.4168367843226278) q[1];
rz(0.0018268213585604415) q[2];
rz(0.09108561063654141) q[3];
rz(1.1822074186046658) q[4];
rz(0.10167094677576481) q[5];
rz(-0.09448786957805654) q[6];
rz(-0.02989885846694781) q[7];
rz(-0.3403870524171752) q[8];
rz(-0.006422646695509965) q[9];
rz(-0.11230442525555877) q[10];
rz(0.051991401006760474) q[11];
cx q[0],q[1];
rz(0.3569158914798906) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.44766641716251154) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.028430768163812913) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.3443369926530416) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(0.9327960528647169) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(0.18958001404971325) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.2001463307911835) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.13958548946480023) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(0.011467403276314885) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.21566013878180418) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(0.04408988296963496) q[11];
cx q[10],q[11];
h q[0];
rz(-0.6322143134853188) q[0];
h q[0];
h q[1];
rz(-0.5798041501162086) q[1];
h q[1];
h q[2];
rz(-1.1899661592823574) q[2];
h q[2];
h q[3];
rz(-0.36886701710514286) q[3];
h q[3];
h q[4];
rz(-0.3913079794804388) q[4];
h q[4];
h q[5];
rz(0.07125876248516949) q[5];
h q[5];
h q[6];
rz(-0.5648746695275609) q[6];
h q[6];
h q[7];
rz(-0.49875545141060973) q[7];
h q[7];
h q[8];
rz(-0.8525596417148245) q[8];
h q[8];
h q[9];
rz(-1.0491760601535811) q[9];
h q[9];
h q[10];
rz(0.2395268491306548) q[10];
h q[10];
h q[11];
rz(-0.08599106005234025) q[11];
h q[11];
rz(0.772845464785728) q[0];
rz(-0.031776230151552495) q[1];
rz(0.002478273992356672) q[2];
rz(0.8809366217620853) q[3];
rz(1.5956324654261709) q[4];
rz(0.21528351812163288) q[5];
rz(0.0546573821239958) q[6];
rz(0.02208020620721146) q[7];
rz(-0.02312048817960084) q[8];
rz(0.01257219441251349) q[9];
rz(0.02865240169577245) q[10];
rz(0.03014433534147701) q[11];
cx q[0],q[1];
rz(1.066621163660084) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(0.7471735387883054) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.03458736795544725) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(-1.1016371302693846) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(0.2568069766710559) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.5671531894152401) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.015761899240081777) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.07334001929180312) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.1849956027514532) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(0.4945846402809996) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(0.013850425268257657) q[11];
cx q[10],q[11];
h q[0];
rz(-0.6356668112742814) q[0];
h q[0];
h q[1];
rz(0.3001521237946691) q[1];
h q[1];
h q[2];
rz(-1.191567581035649) q[2];
h q[2];
h q[3];
rz(-0.16138772498731407) q[3];
h q[3];
h q[4];
rz(-2.072693303203015) q[4];
h q[4];
h q[5];
rz(-0.27390464073723614) q[5];
h q[5];
h q[6];
rz(0.3186622425655053) q[6];
h q[6];
h q[7];
rz(-1.832880552771074) q[7];
h q[7];
h q[8];
rz(-0.5005039331674719) q[8];
h q[8];
h q[9];
rz(-0.1657466866973819) q[9];
h q[9];
h q[10];
rz(-0.7410186741303822) q[10];
h q[10];
h q[11];
rz(-0.1443792192557956) q[11];
h q[11];
rz(0.41822411171318635) q[0];
rz(0.02867906327129984) q[1];
rz(0.002425963038981563) q[2];
rz(-0.5955593500733861) q[3];
rz(-0.1610432966724731) q[4];
rz(0.1574410690385591) q[5];
rz(-0.0349340681783749) q[6];
rz(-0.06018568955203336) q[7];
rz(0.04369128560466298) q[8];
rz(0.00773814250020939) q[9];
rz(0.001582412806925276) q[10];
rz(0.02594272840778414) q[11];
cx q[0],q[1];
rz(0.719642732813619) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.9747614404599885) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.0748676952708909) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(-1.0524721398543415) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(-1.0562943398681366) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.10403796987314844) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.08280903863595653) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.2184922620349364) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.014189292655732764) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.7039026653395863) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(2.2175281670892195) q[11];
cx q[10],q[11];
h q[0];
rz(-1.0971746187463527) q[0];
h q[0];
h q[1];
rz(-1.5293454601318952) q[1];
h q[1];
h q[2];
rz(-1.3469622804364205) q[2];
h q[2];
h q[3];
rz(0.0007843779915194086) q[3];
h q[3];
h q[4];
rz(-0.21680994133313936) q[4];
h q[4];
h q[5];
rz(-0.0851298097835235) q[5];
h q[5];
h q[6];
rz(-0.023764944506221575) q[6];
h q[6];
h q[7];
rz(0.542908939540522) q[7];
h q[7];
h q[8];
rz(0.5413043175630753) q[8];
h q[8];
h q[9];
rz(-0.47959547685793724) q[9];
h q[9];
h q[10];
rz(-1.1647514170598745) q[10];
h q[10];
h q[11];
rz(-1.5188519627353434) q[11];
h q[11];
rz(0.2869164141078826) q[0];
rz(-0.0051306286422414206) q[1];
rz(0.004236342701104893) q[2];
rz(0.4645582846379077) q[3];
rz(0.4370145192757166) q[4];
rz(-0.03367112376352687) q[5];
rz(-0.01734729541382326) q[6];
rz(-0.025360152427020527) q[7];
rz(0.08523066051576389) q[8];
rz(-0.01546704078572708) q[9];
rz(0.029123352583066038) q[10];
rz(0.15351517821398747) q[11];