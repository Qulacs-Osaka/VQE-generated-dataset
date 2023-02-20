OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(2.416672398549541) q[0];
rz(0.7800254284693032) q[0];
ry(2.6014104148737607) q[1];
rz(-0.39010413669398447) q[1];
ry(2.7214674173442135) q[2];
rz(1.8741111440826224) q[2];
ry(0.5968627203046495) q[3];
rz(0.9490693803061083) q[3];
ry(-2.8168610541451686) q[4];
rz(1.115012355247206) q[4];
ry(0.32276374792818435) q[5];
rz(2.5268857096507236) q[5];
ry(1.8848572532558796) q[6];
rz(-1.8740233200698007) q[6];
ry(2.903288396682313) q[7];
rz(-2.1662066393137214) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.730674419198211) q[0];
rz(-0.44262615073086703) q[0];
ry(2.1174399499442345) q[1];
rz(2.5047088243537385) q[1];
ry(-2.13504916284621) q[2];
rz(2.4104750045178336) q[2];
ry(2.0513413124619655) q[3];
rz(1.9295607607089311) q[3];
ry(1.38025099375439) q[4];
rz(-0.7100170988942651) q[4];
ry(-0.7728232576889633) q[5];
rz(-1.0371046738727228) q[5];
ry(-2.1723216588026437) q[6];
rz(2.5730480518721053) q[6];
ry(-1.170068016385128) q[7];
rz(-2.7093094116001937) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.4433982406596817) q[0];
rz(2.4884103900808) q[0];
ry(2.299913023109479) q[1];
rz(1.9631746495117408) q[1];
ry(-0.08343971263321659) q[2];
rz(-1.3203787845311412) q[2];
ry(0.6346775953479078) q[3];
rz(2.624636418866889) q[3];
ry(-2.8467697269120475) q[4];
rz(-1.1905524194511727) q[4];
ry(-0.29303723683422594) q[5];
rz(-2.8188631360683853) q[5];
ry(-0.5581361484675647) q[6];
rz(0.23829022384151965) q[6];
ry(-1.2083517654977287) q[7];
rz(1.5165154880402394) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(3.06920525341299) q[0];
rz(2.642623413860962) q[0];
ry(-1.289893630302152) q[1];
rz(3.140918404719221) q[1];
ry(-2.5523426404783836) q[2];
rz(1.045980114868669) q[2];
ry(-3.004591545308222) q[3];
rz(-0.7232104207386795) q[3];
ry(2.1710470054495676) q[4];
rz(2.822264806356143) q[4];
ry(-2.4251653684946053) q[5];
rz(-1.710814317292563) q[5];
ry(-2.6334805084864534) q[6];
rz(1.3466026965331803) q[6];
ry(0.19434458801982588) q[7];
rz(2.4421225075213675) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.0870338262175587) q[0];
rz(1.7871752361589364) q[0];
ry(-2.506167350693357) q[1];
rz(-1.1715137273540064) q[1];
ry(-2.8870269510014883) q[2];
rz(-2.152091229096751) q[2];
ry(-0.9909530424604887) q[3];
rz(-0.5393071354816055) q[3];
ry(2.7511985005099167) q[4];
rz(1.3050352855590448) q[4];
ry(-2.3255489121285886) q[5];
rz(0.7030778305435039) q[5];
ry(0.6035217496543975) q[6];
rz(-0.8062791573746759) q[6];
ry(1.3925980798330304) q[7];
rz(1.639853580914868) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.1647576502141996) q[0];
rz(2.568974928557224) q[0];
ry(-2.5458617130004915) q[1];
rz(-2.708797435575031) q[1];
ry(-2.648849568608184) q[2];
rz(0.3617778037331796) q[2];
ry(-2.1544001413514655) q[3];
rz(0.15659975940431628) q[3];
ry(2.7101720157454094) q[4];
rz(-2.1702507745193182) q[4];
ry(-2.5949211582655805) q[5];
rz(0.501501358049211) q[5];
ry(3.071501054670598) q[6];
rz(1.9242806585860703) q[6];
ry(-0.8472956724747324) q[7];
rz(1.8509367734400801) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.3163983249669113) q[0];
rz(-2.790675992867789) q[0];
ry(1.9916140045362205) q[1];
rz(-1.7823858276426783) q[1];
ry(-2.884766233445853) q[2];
rz(-2.529998765765896) q[2];
ry(2.840333693438421) q[3];
rz(-0.019216493612185914) q[3];
ry(-0.5463314141511066) q[4];
rz(-0.925697289125112) q[4];
ry(2.450441560525745) q[5];
rz(-2.9494819722732544) q[5];
ry(2.0265353110826263) q[6];
rz(-0.8857132276217264) q[6];
ry(1.5236500167417557) q[7];
rz(-2.2817340519848877) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.206885534891084) q[0];
rz(1.3018392223847606) q[0];
ry(2.916772294684243) q[1];
rz(-1.7680642423915214) q[1];
ry(-0.7887984022270169) q[2];
rz(1.579479310867179) q[2];
ry(-2.3187163595914146) q[3];
rz(1.165844080953165) q[3];
ry(-2.435641943909991) q[4];
rz(-0.8961652538422054) q[4];
ry(-1.8805414612543614) q[5];
rz(0.9537342905568734) q[5];
ry(-0.9200942959986778) q[6];
rz(-0.4521768719996846) q[6];
ry(-1.2562807042790032) q[7];
rz(-2.759723651216011) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.5055899168158255) q[0];
rz(0.9071023943514545) q[0];
ry(-1.2864670312718882) q[1];
rz(-1.4139697919001097) q[1];
ry(1.4935133605443442) q[2];
rz(2.7179944354203958) q[2];
ry(-2.1631920736889687) q[3];
rz(1.279959678979469) q[3];
ry(0.6564915306069192) q[4];
rz(3.001163610184476) q[4];
ry(0.9986710850949729) q[5];
rz(-0.8162727841187788) q[5];
ry(-1.959634448645125) q[6];
rz(-2.489987854635792) q[6];
ry(-0.7940436147784729) q[7];
rz(-2.2912180322801827) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.19887799435080178) q[0];
rz(-2.7789200574870407) q[0];
ry(2.2556481959941936) q[1];
rz(0.3731947600375347) q[1];
ry(0.39293477896605444) q[2];
rz(-1.4177438282052996) q[2];
ry(2.765213308621408) q[3];
rz(2.6644792586036163) q[3];
ry(-2.511539500882146) q[4];
rz(-0.4830541933547181) q[4];
ry(2.64808071051957) q[5];
rz(1.4668403024434973) q[5];
ry(0.20942531806367154) q[6];
rz(1.947298326556684) q[6];
ry(-2.9579318904463734) q[7];
rz(-2.665878523983324) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.35217090729867806) q[0];
rz(0.05274959816490842) q[0];
ry(-2.8282374436703863) q[1];
rz(-0.015859989337706853) q[1];
ry(-0.6876873490616102) q[2];
rz(2.372954931507534) q[2];
ry(-1.3026635231036925) q[3];
rz(-0.5952001121387313) q[3];
ry(-0.4396835380416757) q[4];
rz(-0.8637829740738782) q[4];
ry(-0.8882522338544021) q[5];
rz(0.8192000501456871) q[5];
ry(1.4838657710464016) q[6];
rz(0.2151771126895365) q[6];
ry(-0.5701677698944598) q[7];
rz(-2.540318414815359) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.232308946702192) q[0];
rz(0.9315112563350593) q[0];
ry(2.1869475098757656) q[1];
rz(1.0491354515447113) q[1];
ry(2.5546763927673397) q[2];
rz(1.126782747362145) q[2];
ry(-2.880796689245915) q[3];
rz(0.7553225472193159) q[3];
ry(-0.4747267412511181) q[4];
rz(-0.42363039609631026) q[4];
ry(3.1229879691526845) q[5];
rz(0.43705237385654877) q[5];
ry(2.229576607060851) q[6];
rz(-1.806506347259445) q[6];
ry(1.7693535014784985) q[7];
rz(1.8804012266072494) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.8797799403570108) q[0];
rz(1.6068515928132092) q[0];
ry(0.9366421782415991) q[1];
rz(-0.24690368488085151) q[1];
ry(1.497852407124941) q[2];
rz(-2.46438965966299) q[2];
ry(1.7740083145099685) q[3];
rz(-1.3454621334696126) q[3];
ry(1.1995925076471565) q[4];
rz(2.660277374211309) q[4];
ry(1.063750504572664) q[5];
rz(0.2190167020494814) q[5];
ry(2.2576223114407163) q[6];
rz(2.896881210680728) q[6];
ry(0.4807814817133954) q[7];
rz(-0.2842462913295771) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.1526643278020534) q[0];
rz(-2.908060825500126) q[0];
ry(-0.5703374395222857) q[1];
rz(0.3984801604110718) q[1];
ry(-2.8771053747758164) q[2];
rz(-0.05699575679548145) q[2];
ry(-2.987443667276393) q[3];
rz(-2.4820317161076906) q[3];
ry(-1.1732697634221467) q[4];
rz(-1.1685991410368495) q[4];
ry(-2.7793893232788927) q[5];
rz(-3.115623771166131) q[5];
ry(0.8917623084415676) q[6];
rz(-1.6585009452715307) q[6];
ry(-2.3691602124199314) q[7];
rz(1.0760942691730833) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.6553810538140854) q[0];
rz(1.731266951226961) q[0];
ry(0.7449053691110151) q[1];
rz(1.9685511455896103) q[1];
ry(-0.3410275094571311) q[2];
rz(-2.951107738063475) q[2];
ry(-1.9246491616174932) q[3];
rz(1.3697712623571445) q[3];
ry(0.22328139013957404) q[4];
rz(-1.7332646594735792) q[4];
ry(-0.8481655408326515) q[5];
rz(2.5572978903015877) q[5];
ry(-1.3085199167733013) q[6];
rz(-0.8777829217675474) q[6];
ry(1.5074567940220744) q[7];
rz(1.4468508842147534) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.22316565554196366) q[0];
rz(-1.298404360447754) q[0];
ry(0.23059217820085287) q[1];
rz(-0.6723051425753209) q[1];
ry(-1.6911141973687434) q[2];
rz(2.2446709348298706) q[2];
ry(-1.3561224229196662) q[3];
rz(-0.8189692325042355) q[3];
ry(1.0284478846399383) q[4];
rz(0.08776436234107736) q[4];
ry(1.5866202833792222) q[5];
rz(-1.3576115555779504) q[5];
ry(1.0627231579737613) q[6];
rz(-2.1296914271564975) q[6];
ry(-0.4257946042795471) q[7];
rz(2.4643590918967644) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.3381864677493693) q[0];
rz(-2.306648749686615) q[0];
ry(2.922675739659987) q[1];
rz(-1.8856853450045907) q[1];
ry(-0.6836947277225304) q[2];
rz(2.9851776784317026) q[2];
ry(-2.954964326442908) q[3];
rz(0.7227080631400823) q[3];
ry(-0.5818864616302655) q[4];
rz(2.018033220175573) q[4];
ry(-0.4718180087878799) q[5];
rz(-0.3385720643356977) q[5];
ry(-1.4506609396319652) q[6];
rz(-2.192816241612876) q[6];
ry(0.7483778528178044) q[7];
rz(1.4874064760595649) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.4925278270448086) q[0];
rz(1.2338117360221779) q[0];
ry(-3.0060603857866606) q[1];
rz(-1.8785414680689754) q[1];
ry(3.0966535662619314) q[2];
rz(-2.715729005037114) q[2];
ry(-0.38546059191112736) q[3];
rz(-2.0693711816412375) q[3];
ry(0.4468528400638048) q[4];
rz(-1.2625141620866425) q[4];
ry(-2.7115746729792622) q[5];
rz(-2.9887706626714485) q[5];
ry(1.9704810897433733) q[6];
rz(-0.7339279387628982) q[6];
ry(0.7686436154349) q[7];
rz(2.894436636632302) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.7635847185805652) q[0];
rz(1.3860728217124736) q[0];
ry(-2.8050183902802717) q[1];
rz(1.8269717956603901) q[1];
ry(0.7772123859320956) q[2];
rz(1.488940541029426) q[2];
ry(2.6529027490036614) q[3];
rz(-3.0345737927383953) q[3];
ry(0.716957759058209) q[4];
rz(-1.2343094425288008) q[4];
ry(-2.0388127736921513) q[5];
rz(2.7250464326364043) q[5];
ry(-0.16706044947299772) q[6];
rz(-3.0889725925969937) q[6];
ry(1.6654913528499957) q[7];
rz(2.871723250215417) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.590707084431551) q[0];
rz(-2.9190520260263466) q[0];
ry(-2.006438157642201) q[1];
rz(-0.07726361723450464) q[1];
ry(3.0362358836108774) q[2];
rz(-1.156989441537581) q[2];
ry(-0.2791925016165652) q[3];
rz(0.33475363343288933) q[3];
ry(1.9965924169841154) q[4];
rz(2.696676936822453) q[4];
ry(-0.17663256318480236) q[5];
rz(1.7690868212268591) q[5];
ry(-1.119093436962616) q[6];
rz(-3.002379601543674) q[6];
ry(-1.237738881109093) q[7];
rz(-0.003443016054663981) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.730296296465057) q[0];
rz(-0.005816099932196925) q[0];
ry(2.7722790442416265) q[1];
rz(-1.5407851007074254) q[1];
ry(1.243457833661843) q[2];
rz(2.891833183302905) q[2];
ry(-0.14066234037862957) q[3];
rz(1.423236304477114) q[3];
ry(-1.8297360380082897) q[4];
rz(-1.3489626640191825) q[4];
ry(2.3328271189983756) q[5];
rz(-1.579844629148306) q[5];
ry(0.5489774539812506) q[6];
rz(2.247933666418772) q[6];
ry(-0.9955752826607602) q[7];
rz(-1.6083410053434406) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.9277248426872093) q[0];
rz(-0.3539129834739907) q[0];
ry(-1.758362134853946) q[1];
rz(1.0294557805570905) q[1];
ry(-2.550114588636207) q[2];
rz(2.9257446994565544) q[2];
ry(-2.0353336232895023) q[3];
rz(-1.7949425714046627) q[3];
ry(2.9277347296330105) q[4];
rz(-1.8005804470949363) q[4];
ry(0.664556928695033) q[5];
rz(-1.7315109907637245) q[5];
ry(0.9951781083419817) q[6];
rz(2.9555326576945045) q[6];
ry(2.473582580057767) q[7];
rz(-1.0335376158463148) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.9213778115857507) q[0];
rz(-1.2981912995810672) q[0];
ry(0.1713469980668867) q[1];
rz(-1.851679446561997) q[1];
ry(-0.6279836184544553) q[2];
rz(-0.3885485670541575) q[2];
ry(1.1093931923598788) q[3];
rz(-0.5946709708226257) q[3];
ry(-1.8862202949614177) q[4];
rz(-2.61508605520741) q[4];
ry(-0.16497217543251352) q[5];
rz(1.1825162979152868) q[5];
ry(-2.460970742194631) q[6];
rz(-0.7991423034726585) q[6];
ry(0.14381615169158843) q[7];
rz(0.09496348265307922) q[7];