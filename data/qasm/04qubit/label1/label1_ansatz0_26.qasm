OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.034514646734394874) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.13010097968139012) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.022319710659776525) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.23014961373471618) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.12131282755585679) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.056044799447228615) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.10538353090112133) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.03039509540590487) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.10118943904432867) q[3];
cx q[2],q[3];
rz(-0.053919663264368936) q[0];
rz(-0.06406228858291674) q[1];
rz(-0.0809316297528058) q[2];
rz(-0.06474956062945554) q[3];
rx(-0.09840238397732674) q[0];
rx(-0.12627089481091056) q[1];
rx(0.017465621300644035) q[2];
rx(-0.19116397194266221) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.02152283956990881) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.0864130013374324) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.03131429617783068) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.17101019749290253) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.03463094807996398) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.057125169329931454) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.1512952000815811) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.002899944845854779) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.12695561198841146) q[3];
cx q[2],q[3];
rz(-0.026525481666385894) q[0];
rz(-0.09862735926580181) q[1];
rz(-0.10376909998574867) q[2];
rz(-0.035940040202191235) q[3];
rx(-0.06727419699600472) q[0];
rx(-0.19426315594616436) q[1];
rx(0.04315375448143868) q[2];
rx(-0.27005520923257526) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.03673276360321155) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.17266425227497684) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.10238167152938941) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.09277156703267712) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.00018153135784188703) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.049705687933794575) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.16746684727852848) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.08355542853923452) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.050469131913050934) q[3];
cx q[2],q[3];
rz(-0.021310052524578706) q[0];
rz(-0.08242622173255038) q[1];
rz(-0.08933925532891018) q[2];
rz(-0.11702691047668498) q[3];
rx(-0.08822895054888187) q[0];
rx(-0.19666057078991364) q[1];
rx(0.019329061832299408) q[2];
rx(-0.25567604114812453) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.007530526579511833) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.1692457185846358) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.06846047442618768) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.05182919612663239) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.07142613344404525) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.09588179388074647) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.08685557482096952) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.055520851133739835) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.0794895711216715) q[3];
cx q[2],q[3];
rz(-0.04316238958133229) q[0];
rz(-0.07916313836316462) q[1];
rz(0.0726390062120821) q[2];
rz(-0.1053801920627978) q[3];
rx(-0.14957855845568246) q[0];
rx(-0.2929185704188215) q[1];
rx(0.06354010547541335) q[2];
rx(-0.23268324411255847) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.0170968911520464) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.10618227168278331) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.03805222659584248) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.009251707950547392) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.05769862991651998) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.10374567585788116) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.017422488951680196) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.09378625370362237) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.04031415060994611) q[3];
cx q[2],q[3];
rz(-0.045870613938106305) q[0];
rz(-0.019443474219683697) q[1];
rz(0.13456818269027837) q[2];
rz(-0.11652931472669523) q[3];
rx(-0.1253825794888658) q[0];
rx(-0.3143083440519809) q[1];
rx(0.05898540113334841) q[2];
rx(-0.20244618902219036) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.03174398523792075) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.07420895726429323) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.060265396017421737) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.08590714697754097) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.08929853006644511) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.05475685354818906) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.037377911387218014) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.1450334611402293) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.13270726858986107) q[3];
cx q[2],q[3];
rz(-0.013090850045746068) q[0];
rz(0.02659571996310632) q[1];
rz(0.20234958922820223) q[2];
rz(-0.1334793804207082) q[3];
rx(-0.18653250486109038) q[0];
rx(-0.356549531408855) q[1];
rx(0.02354650987439782) q[2];
rx(-0.21427047924423934) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.045479307510294276) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.12172635783489225) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.06718586208959099) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.13071585777149244) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.039463548313230117) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.008580422239237584) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.007319225002753004) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.10225769484137595) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.10303492681828806) q[3];
cx q[2],q[3];
rz(-0.005866150002022761) q[0];
rz(-0.041204482777938096) q[1];
rz(0.24383470592149648) q[2];
rz(-0.14094423676592346) q[3];
rx(-0.2017607562501459) q[0];
rx(-0.33364606628449917) q[1];
rx(-0.03507512777205782) q[2];
rx(-0.1934779014448751) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.019766537675529532) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.0936801194554654) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.02176147476407819) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.08762098032292301) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.10692646628076444) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.054791969303320864) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.01418162195090317) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.1973174666490822) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.06784980351633606) q[3];
cx q[2],q[3];
rz(0.06429891358153492) q[0];
rz(0.027347526248683814) q[1];
rz(0.3158665501297237) q[2];
rz(-0.04600374831147095) q[3];
rx(-0.15518256897611563) q[0];
rx(-0.2766581184273364) q[1];
rx(-0.08900706550566466) q[2];
rx(-0.27029641902996415) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.047132623705860695) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.07715863271552773) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.08114034383605864) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.03666173960272704) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.08804338728716715) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.06902454276060871) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.08250906892067335) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.1552864564622148) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.01430202389055493) q[3];
cx q[2],q[3];
rz(0.02092315114786015) q[0];
rz(-0.005487966349797388) q[1];
rz(0.24947419772234333) q[2];
rz(-0.06849389778444549) q[3];
rx(-0.18652166154033847) q[0];
rx(-0.24759858411540947) q[1];
rx(-0.16007753019192264) q[2];
rx(-0.2487520208606315) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.03249479057701199) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.05324695333091054) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.09069986273149137) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.08242596462526985) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.0644420047515763) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.005088433061516078) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.06492949397835204) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.17408067512239025) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.02702050304602101) q[3];
cx q[2],q[3];
rz(0.025836872111867953) q[0];
rz(-0.005434042786726184) q[1];
rz(0.175921664484132) q[2];
rz(0.0043404201522486245) q[3];
rx(-0.20147218248759105) q[0];
rx(-0.22361007871921362) q[1];
rx(-0.1475858968460866) q[2];
rx(-0.24079792500086886) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.02127746812616631) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.0345363468487324) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.018349081127376963) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.004483285080927773) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.06668564422733994) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.05738569431546859) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.18716343124202386) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.16937190212642367) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.07262229542187634) q[3];
cx q[2],q[3];
rz(0.02233923452667442) q[0];
rz(0.0758584668665022) q[1];
rz(0.16924378520911987) q[2];
rz(-0.03635306778108573) q[3];
rx(-0.24932254157330858) q[0];
rx(-0.2552479148448094) q[1];
rx(-0.23473510326752386) q[2];
rx(-0.2319344211491905) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.03479223756231022) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.04958484074571153) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.05093467063354841) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.02419888023482608) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.04632618593681242) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.04094909547221821) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.17963694905328642) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.15792121349761365) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.05536585822925394) q[3];
cx q[2],q[3];
rz(0.021461257046066433) q[0];
rz(0.09970087806977589) q[1];
rz(0.1064700678840885) q[2];
rz(-0.009463979789834672) q[3];
rx(-0.24412652485587244) q[0];
rx(-0.2406907463006181) q[1];
rx(-0.22790321438098826) q[2];
rx(-0.24719613678163666) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.011816423601586301) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.009895927058622236) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.11728977327848483) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.07827666508958901) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.07311317559722905) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.11237244122871887) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.17394523596000613) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.17530521829106135) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.10738956565185763) q[3];
cx q[2],q[3];
rz(0.08713691963886395) q[0];
rz(0.04432308609437402) q[1];
rz(0.03688381067244494) q[2];
rz(-0.01497311760452532) q[3];
rx(-0.2126224520496586) q[0];
rx(-0.2552433457238348) q[1];
rx(-0.14826181960572005) q[2];
rx(-0.21677871471454307) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.05003942406842453) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.03511516464001753) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.13372183329402346) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.09582536744017842) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.05172145537874085) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.0811305761483584) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.1463430906174416) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.18076749617513577) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.0870481289325185) q[3];
cx q[2],q[3];
rz(0.009764904893751482) q[0];
rz(0.047096010344347905) q[1];
rz(-0.04280587151998624) q[2];
rz(0.02929313593593386) q[3];
rx(-0.23831542030753464) q[0];
rx(-0.1637035066310168) q[1];
rx(-0.2172918528001449) q[2];
rx(-0.15723955043841192) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.03393391780986073) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.0925937125823583) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.18247946795944295) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.14097394000174318) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.07116101358449087) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.20452283946864414) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.20346827429928835) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.11780252817747032) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.0025094200640261564) q[3];
cx q[2],q[3];
rz(0.01421110856592293) q[0];
rz(0.00578417803450191) q[1];
rz(-0.13966388652736814) q[2];
rz(0.07870034452063975) q[3];
rx(-0.22473904446802553) q[0];
rx(-0.2106771425345278) q[1];
rx(-0.17804594104763363) q[2];
rx(-0.17901012546029188) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.11855394135026796) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.04550717831276798) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.20325712131390047) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.23028750030627454) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.1116163889207379) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.17110710464735562) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.19509497937817397) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.1644045079874102) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.07636620163790148) q[3];
cx q[2],q[3];
rz(0.03330319526519297) q[0];
rz(-0.08875156227743314) q[1];
rz(-0.15500848585030663) q[2];
rz(0.07807322236050104) q[3];
rx(-0.32322701686054095) q[0];
rx(-0.1835758755603984) q[1];
rx(-0.14379813818472129) q[2];
rx(-0.26813000517931185) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.18948005929241613) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.08514525533228073) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.15261777237704488) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.22973720240745976) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.11323967379232863) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.20770375675455194) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.16592847250425755) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.08225185602190097) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.09406244424492835) q[3];
cx q[2],q[3];
rz(-0.011228516688698299) q[0];
rz(-0.1285648068295131) q[1];
rz(-0.15415186279955095) q[2];
rz(0.07547664778532939) q[3];
rx(-0.3044992989046171) q[0];
rx(-0.14805637664985336) q[1];
rx(-0.17273241112438575) q[2];
rx(-0.22620633403977392) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.17333933373704816) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.117319727545731) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.1952094751901867) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.296533269595129) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.11807403842537573) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.19792198780829995) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.13038797083294992) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.01745083504804569) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.13846774121638342) q[3];
cx q[2],q[3];
rz(-0.08635409960943265) q[0];
rz(-0.17860111397011427) q[1];
rz(-0.08483089667702974) q[2];
rz(0.0358353124729554) q[3];
rx(-0.27534009990903974) q[0];
rx(-0.11742611834143826) q[1];
rx(-0.17699053445996615) q[2];
rx(-0.2843504666090575) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.2158813721165169) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.10871820362370399) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.20133906665422674) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.2175920036657867) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.06725221509613867) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.16043847295689082) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.14109585050744447) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.122398138639085) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.07196171832263298) q[3];
cx q[2],q[3];
rz(-0.11551121608619166) q[0];
rz(-0.12201929118091796) q[1];
rz(-0.017706869294995587) q[2];
rz(0.03293614658326901) q[3];
rx(-0.2788212263065776) q[0];
rx(-0.11620244617431877) q[1];
rx(-0.22195961395304045) q[2];
rx(-0.24605565940441146) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.26495221665348634) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.15082765186687017) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.18419907923263346) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.28219473610368556) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.003133884082962287) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.056632239993879514) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.1471178055138065) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.15795974328403467) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.06551618966075942) q[3];
cx q[2],q[3];
rz(-0.16603860731888911) q[0];
rz(-0.11887791477295728) q[1];
rz(0.09726645973860679) q[2];
rz(0.016590022887948988) q[3];
rx(-0.29107491252381296) q[0];
rx(-0.07131340075452193) q[1];
rx(-0.26008098831536447) q[2];
rx(-0.1755017467234539) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.21877091135586096) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.19972731420226725) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.09063715547379436) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.21651592903173705) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.04880784628624385) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.007226360250826966) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.1551797033553536) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.11127035441952486) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.06365513068496503) q[3];
cx q[2],q[3];
rz(-0.2052006897473019) q[0];
rz(-0.11141701531778422) q[1];
rz(0.1443882709059352) q[2];
rz(-0.032062727544584904) q[3];
rx(-0.289231134544576) q[0];
rx(-0.041196773274793604) q[1];
rx(-0.2223686504253166) q[2];
rx(-0.14826845397482918) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.2517435620066628) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.1345076044196851) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.0061740198897177285) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.30933548763918517) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.04990804736846325) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.04838165962401572) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.11532134199131296) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.07704642012385293) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.0082799206824966) q[3];
cx q[2],q[3];
rz(-0.2091460678376667) q[0];
rz(-0.08690945810105485) q[1];
rz(0.11735806678307248) q[2];
rz(-0.08780603742162522) q[3];
rx(-0.22671703553499184) q[0];
rx(-0.06999402122225405) q[1];
rx(-0.1523019495361107) q[2];
rx(-0.14017021615882658) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.31241720569368564) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.1611482351194566) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.12695408940935599) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.28684009144131445) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.04140473377977897) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.033738827222873684) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.1280147440192477) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.17087754393369936) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.007790951274767091) q[3];
cx q[2],q[3];
rz(-0.23453781326279938) q[0];
rz(-0.012428861205099012) q[1];
rz(0.023218176076024815) q[2];
rz(-0.0329870294458335) q[3];
rx(-0.28608304514551697) q[0];
rx(-0.001678678352752502) q[1];
rx(-0.20065875969737607) q[2];
rx(-0.1948360073403587) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.28180199897177116) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.09862999969602569) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.04445513804727431) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.33622936424393884) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.07946047890437158) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.025879366003093995) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.0350585849899437) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.1944766797929061) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.03139893056917122) q[3];
cx q[2],q[3];
rz(-0.22065859374105282) q[0];
rz(0.0750361465802567) q[1];
rz(0.002304181448224962) q[2];
rz(-0.074010183088598) q[3];
rx(-0.19739575320130082) q[0];
rx(-0.0502095855943429) q[1];
rx(-0.13780217930097463) q[2];
rx(-0.23151798873336774) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.2476362858221786) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.08688603520604407) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.02416287304044766) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.35288446371579363) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.14074682460656032) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.04283279795458315) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.0470708214261417) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.24429085879354973) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.020011716204782357) q[3];
cx q[2],q[3];
rz(-0.25102348986050227) q[0];
rz(0.08095702640648092) q[1];
rz(-0.04201679051082337) q[2];
rz(-0.0661292663258542) q[3];
rx(-0.20414464059890866) q[0];
rx(-0.10017366864808509) q[1];
rx(-0.16892401655713285) q[2];
rx(-0.22182476838526166) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.2368786749850805) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.24490107750373222) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.13391496805568157) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.34116187954055543) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.2303542851329162) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.1133463276928007) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.004804276812106918) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.14267306822119194) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.06353451396397122) q[3];
cx q[2],q[3];
rz(-0.2550376288432288) q[0];
rz(0.11920889235553737) q[1];
rz(-0.021366901858560345) q[2];
rz(-0.1386597666726269) q[3];
rx(-0.12612371763515118) q[0];
rx(-0.12830062689485344) q[1];
rx(-0.07046236072007865) q[2];
rx(-0.12733172941200963) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.23961487204978074) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.27195872467004656) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.1197101879721682) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.297903675753574) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.2583166901469849) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.013783473507723838) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.09273292265842074) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.14550849416738534) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.12219943484437958) q[3];
cx q[2],q[3];
rz(-0.30892523399928046) q[0];
rz(0.19687274911192332) q[1];
rz(0.020136514317763525) q[2];
rz(-0.17776589065349427) q[3];
rx(-0.125014206485037) q[0];
rx(-0.13817840921490165) q[1];
rx(-0.05853149706326293) q[2];
rx(-0.08978540600394369) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.17929988248057635) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.17599177295022336) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.08268480794352195) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.34657952847065293) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.1376977760121746) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.1309312118849243) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.05638404049206249) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.07851247800452034) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.1425895248827624) q[3];
cx q[2],q[3];
rz(-0.27109965845548256) q[0];
rz(0.3552539767708588) q[1];
rz(-0.03785394286905237) q[2];
rz(-0.20509940416229633) q[3];
rx(-0.03992571062237422) q[0];
rx(-0.19450735172680547) q[1];
rx(-0.11231107677872412) q[2];
rx(-0.12136786796672569) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.06770971440223826) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.1775601279607454) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.05647150074089286) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.15516620884578203) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.10517633788778287) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.10433103537743972) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.07180329848804608) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.04726358167394468) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.11396755298869249) q[3];
cx q[2],q[3];
rz(-0.3006327646816106) q[0];
rz(0.41665360406752117) q[1];
rz(-0.039551645487960904) q[2];
rz(-0.25963634741578523) q[3];
rx(-0.10782194500814436) q[0];
rx(-0.2156194341669064) q[1];
rx(0.013229914321289424) q[2];
rx(-0.05039240779285262) q[3];