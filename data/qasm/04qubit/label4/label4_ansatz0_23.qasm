OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.0379408779161158) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.05312603483001994) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.0011171494010963577) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.08033877379738505) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.04107052887890265) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.05431179993821374) q[3];
cx q[2],q[3];
rx(-0.05283638856303757) q[0];
rz(-0.04911083509338916) q[0];
rx(0.007984228888894494) q[1];
rz(-0.011587899296900312) q[1];
rx(-0.02548028779811427) q[2];
rz(-0.03331668438812298) q[2];
rx(-0.11391364910753153) q[3];
rz(-0.08651587131795181) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.030271587431012915) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.0011769992936888228) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.05855554580499469) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.0790111347165578) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.011562370210428961) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.04513443114170818) q[3];
cx q[2],q[3];
rx(-0.09628005258288236) q[0];
rz(-0.057534168430860164) q[0];
rx(-0.014151292856029929) q[1];
rz(-0.05719805851893796) q[1];
rx(-0.009305505517629976) q[2];
rz(-0.1013216357349551) q[2];
rx(-0.0837901188184789) q[3];
rz(-0.0760237432450995) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.07170773110978192) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.01856052119796612) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.08985117096232309) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.07319835475484841) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.01830854973512462) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.009389497087262777) q[3];
cx q[2],q[3];
rx(-0.13349653758629748) q[0];
rz(-0.07251887524246935) q[0];
rx(0.020224865434149565) q[1];
rz(-0.10982352954975895) q[1];
rx(-0.024562331317269894) q[2];
rz(-0.11200420152260392) q[2];
rx(-0.13743526594127878) q[3];
rz(-0.039482987302402114) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.14984720846354996) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.004880930726088067) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.044991113452095534) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.0722362720802805) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.04584995903051591) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.058652912287679125) q[3];
cx q[2],q[3];
rx(-0.07940516973694343) q[0];
rz(0.003006347924127728) q[0];
rx(-0.02634738716150187) q[1];
rz(-0.10840436070822962) q[1];
rx(-0.09150982106680708) q[2];
rz(-0.1222462434960371) q[2];
rx(-0.1260958388781751) q[3];
rz(-0.037537073298745505) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.1409455465764644) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.03740522266245714) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.04304769574604645) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.13595396923886977) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.03318173281608664) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.004574438845730172) q[3];
cx q[2],q[3];
rx(-0.09506267782263735) q[0];
rz(-0.012663058468651405) q[0];
rx(-0.025504251322742817) q[1];
rz(-0.11464817503144296) q[1];
rx(-0.031116854760854115) q[2];
rz(-0.08300113695394744) q[2];
rx(-0.12437501787936622) q[3];
rz(-0.024452257564369438) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.12450398164497015) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.028538471383928943) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.0008823229267994149) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.0629007806631311) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.019892928449434216) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.07544630270121656) q[3];
cx q[2],q[3];
rx(-0.08093448296139952) q[0];
rz(-0.04469855734552219) q[0];
rx(-0.02247206790610122) q[1];
rz(-0.04100512275042899) q[1];
rx(-0.026010425360275872) q[2];
rz(-0.08047018285950594) q[2];
rx(-0.11520187854610171) q[3];
rz(-0.08967606583175851) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.12986963283011882) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.003738433502335526) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.00802812610717791) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.13483775355854322) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.026771780586251993) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.034168426034806534) q[3];
cx q[2],q[3];
rx(-0.1059527772720859) q[0];
rz(-0.05327647232601443) q[0];
rx(0.0036430816881776345) q[1];
rz(-0.031331522973458543) q[1];
rx(-0.07349853929321953) q[2];
rz(-0.1282699578320449) q[2];
rx(-0.03997608684887642) q[3];
rz(-0.09579922322825704) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.1481312509096359) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.05326098066539924) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.03181318948560882) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.11467170176919828) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.014906050751960928) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.030851998721360026) q[3];
cx q[2],q[3];
rx(-0.07866012724467317) q[0];
rz(-0.07652339987328106) q[0];
rx(-0.07479777448588988) q[1];
rz(-0.11181675209623222) q[1];
rx(-0.023447487238237318) q[2];
rz(-0.12858091789023562) q[2];
rx(-0.11965831904040598) q[3];
rz(-0.007329691761508896) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.13646143543395856) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.02324723041758844) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.006639631022478296) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.1334575088992955) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.025698116254447883) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.018222477049559104) q[3];
cx q[2],q[3];
rx(-0.0944956288177432) q[0];
rz(-0.01995206206356799) q[0];
rx(-0.02412428430513006) q[1];
rz(-0.08287220492498545) q[1];
rx(0.010530917790068718) q[2];
rz(-0.1190978505697892) q[2];
rx(-0.09770979184527075) q[3];
rz(-0.045446460694307224) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.09457249373543569) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.021411626135561454) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.0698515654203648) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.15987946030774888) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.0299283334065873) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.00979507825747074) q[3];
cx q[2],q[3];
rx(-0.1501801750816394) q[0];
rz(-0.07752568253759344) q[0];
rx(-0.03821545143821277) q[1];
rz(-0.11797547275791759) q[1];
rx(0.011780294868450848) q[2];
rz(-0.06498798313419288) q[2];
rx(-0.06460115985707554) q[3];
rz(-0.0044996454913928344) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.07618342072148294) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.0004896914601021375) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.007645060255910277) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.10465605711282902) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.034863633477017965) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.02912369392854666) q[3];
cx q[2],q[3];
rx(-0.14192136595516092) q[0];
rz(-0.029570024383297693) q[0];
rx(-0.010927257523494268) q[1];
rz(-0.039934037839742614) q[1];
rx(0.037873716365097274) q[2];
rz(-0.11263918939013313) q[2];
rx(-0.059058373470185305) q[3];
rz(-0.05565370110823109) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.06544014214621882) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.03855829157177547) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.04528217682209338) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.1454822159638549) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.04790608816241719) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.026081710943681502) q[3];
cx q[2],q[3];
rx(-0.15286812943539885) q[0];
rz(-0.026059950808536084) q[0];
rx(-0.0019640294048049074) q[1];
rz(-0.06861166899522432) q[1];
rx(0.036676869873253276) q[2];
rz(-0.0781432803658834) q[2];
rx(-0.04206920907207893) q[3];
rz(0.006483648299479633) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.1421084026449311) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.03253352258107258) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.02699209133635817) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.08543715147172917) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.01216276962552126) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.08175703703083428) q[3];
cx q[2],q[3];
rx(-0.1073914118032736) q[0];
rz(-0.024574032003617723) q[0];
rx(0.0030383240787156232) q[1];
rz(-0.06690611535666158) q[1];
rx(-0.010097503190415148) q[2];
rz(-0.05562687363456672) q[2];
rx(-0.09546679953064645) q[3];
rz(-0.014667918649572106) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.17914603066213547) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.004836709500612044) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.048296862948087466) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.06407404911809124) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.0627429191038834) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.06689465113887279) q[3];
cx q[2],q[3];
rx(-0.16393678347088628) q[0];
rz(0.01234576938816812) q[0];
rx(0.025936359069816655) q[1];
rz(-0.026932845425055536) q[1];
rx(0.0352571835394975) q[2];
rz(-0.11402723488539344) q[2];
rx(-0.11824650838136772) q[3];
rz(0.02547661665185103) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.11650644072439248) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.03739998725447676) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.022142242224831455) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.06446312648493345) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.003654868275506394) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.04074789107516001) q[3];
cx q[2],q[3];
rx(-0.12336190713763599) q[0];
rz(0.02508480821558947) q[0];
rx(-0.005538663286281544) q[1];
rz(-0.05752780054452829) q[1];
rx(-0.0425708933264931) q[2];
rz(-0.026427457466306613) q[2];
rx(-0.1551871261315474) q[3];
rz(0.005700932361708602) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.08200529240920683) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.01824882920953679) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.04169217001477705) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.011944125609233523) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.02160884987013611) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.03160029806688249) q[3];
cx q[2],q[3];
rx(-0.1572012517604868) q[0];
rz(0.04332500488654301) q[0];
rx(0.01458851389340082) q[1];
rz(-0.06179316526852159) q[1];
rx(0.021623473346236433) q[2];
rz(-0.02552442976158919) q[2];
rx(-0.17616829248451116) q[3];
rz(-0.0015420962856246153) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.04324286462573848) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.03573130250754135) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-5.0904663685582795e-05) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.010580563279970014) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.016876937761544333) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.03138824949775532) q[3];
cx q[2],q[3];
rx(-0.19109362843454442) q[0];
rz(-0.02212880994445429) q[0];
rx(0.044617307920467836) q[1];
rz(-0.12051774148210308) q[1];
rx(-0.0045726556147925625) q[2];
rz(-0.09615342292618105) q[2];
rx(-0.1697726763172676) q[3];
rz(0.040826350497830885) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.11671593985310873) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.01929576393950388) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.034872960051840915) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.07386911692900862) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.02888809631005893) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.017988303613169533) q[3];
cx q[2],q[3];
rx(-0.1535210391807772) q[0];
rz(-0.020543213166026492) q[0];
rx(0.016966486273137845) q[1];
rz(-0.05980629928874775) q[1];
rx(0.009344440289379793) q[2];
rz(-0.10907329548992678) q[2];
rx(-0.18369861020456624) q[3];
rz(-0.0030715826452740436) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.07109971712999291) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.05781516861282034) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.026115274907546903) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.021018101698891133) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.009796277257825621) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.02156518467463126) q[3];
cx q[2],q[3];
rx(-0.1420344629923179) q[0];
rz(0.01205358888429394) q[0];
rx(-0.04128822925645619) q[1];
rz(-0.11176375739662353) q[1];
rx(0.011287370907147916) q[2];
rz(-0.11564246624071627) q[2];
rx(-0.12183380855565057) q[3];
rz(0.05672642926091883) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.1497319831586052) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.013662541699580392) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.037607969100896935) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.03466140002977103) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.04935033165432327) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.04797306574105032) q[3];
cx q[2],q[3];
rx(-0.16264077064606666) q[0];
rz(-0.012404988684701555) q[0];
rx(-0.052668277912240595) q[1];
rz(-0.13994145361216362) q[1];
rx(-0.05271453888333901) q[2];
rz(-0.10971263263887686) q[2];
rx(-0.15912741131097322) q[3];
rz(0.052697121178768964) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.11276381864248029) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.09981062230469379) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.04333011677444247) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.14549946842431372) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.12697150883701158) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.0019043455927301874) q[3];
cx q[2],q[3];
rx(-0.13149922771062117) q[0];
rz(0.002738995167961967) q[0];
rx(-0.0071713454755776606) q[1];
rz(-0.07768415874961) q[1];
rx(0.01048861598546409) q[2];
rz(-0.09636075596278999) q[2];
rx(-0.1392648666251156) q[3];
rz(0.02751390561680815) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.16118361222137645) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.12483646063611703) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.00837099785154436) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.14784073668650297) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.12847870617679785) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.013555853821609828) q[3];
cx q[2],q[3];
rx(-0.1268730582302866) q[0];
rz(-0.008320594934711606) q[0];
rx(0.034011635498659425) q[1];
rz(-0.056603046436987034) q[1];
rx(0.009205617987557146) q[2];
rz(-0.08643920255071544) q[2];
rx(-0.15844371145561012) q[3];
rz(-0.016134680680276813) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.15686806529178235) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.10562841935601239) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.06852498442662476) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.06219950572006346) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.038433940239375484) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.07425460751192822) q[3];
cx q[2],q[3];
rx(-0.10291526711636498) q[0];
rz(-0.057950000634581667) q[0];
rx(-0.02117166301662213) q[1];
rz(-0.0890860207921048) q[1];
rx(-0.04792545988449197) q[2];
rz(-0.06321300052032283) q[2];
rx(-0.16244528397133576) q[3];
rz(-0.05957165801690644) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.13416724659378695) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.07032346795376382) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.004179868926141901) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.14955047215553904) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.09635970857910056) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.007335991670894288) q[3];
cx q[2],q[3];
rx(-0.10747228331739046) q[0];
rz(-0.04653092255363346) q[0];
rx(-0.026192359433239897) q[1];
rz(-0.03712347160604578) q[1];
rx(0.0524742954477149) q[2];
rz(-0.08567533492425139) q[2];
rx(-0.14045893600879958) q[3];
rz(-0.06647220843480726) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.07957119303608434) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.06065096226706645) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.016572020302985203) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.06557151031295554) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.005359611641262665) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.041548660314583) q[3];
cx q[2],q[3];
rx(-0.09921531787790693) q[0];
rz(-0.09201639170604481) q[0];
rx(0.04498996447502691) q[1];
rz(-0.07388109853214878) q[1];
rx(-0.052578744784648794) q[2];
rz(-0.08164389398379156) q[2];
rx(-0.12089532569216267) q[3];
rz(-0.05492298707783455) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.041794075005223065) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.04109079781916449) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.05387094780631836) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.0317010901522617) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.04936932771874921) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.010794794861244687) q[3];
cx q[2],q[3];
rx(-0.1577681129758365) q[0];
rz(-0.04966798223617184) q[0];
rx(-0.04561699052941094) q[1];
rz(-0.09120872562098162) q[1];
rx(0.01439125922708741) q[2];
rz(-0.05471612738586314) q[2];
rx(-0.18918409062677005) q[3];
rz(-0.0027214778667177556) q[3];