OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-1.248113641767927) q[0];
rz(-2.8593304892684657) q[0];
ry(1.2515774857872763) q[1];
rz(2.74427631872346) q[1];
ry(2.3175640897806136) q[2];
rz(-2.386443560640035) q[2];
ry(1.4244861284818582) q[3];
rz(0.41355318066050817) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.9387900642913921) q[0];
rz(-2.301642001685129) q[0];
ry(-0.5379824130194771) q[1];
rz(-2.5354915557944664) q[1];
ry(0.16848234490903202) q[2];
rz(-2.387452206090332) q[2];
ry(-0.7382301448298794) q[3];
rz(2.6936936609463045) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.8170779819644208) q[0];
rz(0.44278982399624667) q[0];
ry(-2.801594772420005) q[1];
rz(-0.7331093466769806) q[1];
ry(-0.23706921637380005) q[2];
rz(-3.0901673421268243) q[2];
ry(0.30796093289736604) q[3];
rz(0.4463573285832316) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.268458385263127) q[0];
rz(-2.4588644872020238) q[0];
ry(0.10040005195437018) q[1];
rz(-2.890616866537093) q[1];
ry(-2.176339189872694) q[2];
rz(-0.41594337994234115) q[2];
ry(2.1379335549277867) q[3];
rz(-2.6489507895646516) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.617217124730185) q[0];
rz(-1.9974390991662616) q[0];
ry(-2.784357508781214) q[1];
rz(-0.5695991110925558) q[1];
ry(1.349760503951074) q[2];
rz(0.5611330189597892) q[2];
ry(-2.1694866377945337) q[3];
rz(2.2094628271298067) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.0276596584668227) q[0];
rz(-0.16812360862866438) q[0];
ry(1.4853221030629085) q[1];
rz(2.350465879673152) q[1];
ry(3.1092818053152715) q[2];
rz(-3.132588643285292) q[2];
ry(-1.979627082108589) q[3];
rz(-2.767904098388421) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.5838009540353836) q[0];
rz(-3.0827771528699572) q[0];
ry(-2.987539093133734) q[1];
rz(-0.41405741828999026) q[1];
ry(2.29129623687896) q[2];
rz(-0.7263741790686367) q[2];
ry(3.089949199185843) q[3];
rz(-1.5276867124934352) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.379879190178702) q[0];
rz(-0.48598981034258415) q[0];
ry(-2.316141838992265) q[1];
rz(-0.6618918170295912) q[1];
ry(-2.6977919757432485) q[2];
rz(1.6805989898388871) q[2];
ry(-1.3327771802602149) q[3];
rz(-0.05070921716940748) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.3216329074893203) q[0];
rz(-0.6306775263432295) q[0];
ry(0.6209728171036701) q[1];
rz(2.10310630739478) q[1];
ry(-2.8924813020763707) q[2];
rz(2.8561886419882487) q[2];
ry(0.1889473490886724) q[3];
rz(-2.320829874248227) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.0436263079142334) q[0];
rz(-0.6036250810378931) q[0];
ry(-0.5797486994522486) q[1];
rz(-2.6209210237504323) q[1];
ry(-0.7578486187862096) q[2];
rz(-0.8238524043186403) q[2];
ry(-1.4133909230778883) q[3];
rz(1.010668433583304) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.7121085820131271) q[0];
rz(-2.3770762902739007) q[0];
ry(-0.8168706269754652) q[1];
rz(-0.1302353306963281) q[1];
ry(-0.452809197414667) q[2];
rz(-2.1768044225901364) q[2];
ry(1.634948566985092) q[3];
rz(1.9064495183079988) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.8970136695751275) q[0];
rz(-2.8436566817594144) q[0];
ry(1.225771219811091) q[1];
rz(2.43869189569187) q[1];
ry(1.831431359769377) q[2];
rz(1.9461098169079865) q[2];
ry(2.3111603516412327) q[3];
rz(-3.0578081488229847) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.0244972578650442) q[0];
rz(-1.4328850637958293) q[0];
ry(0.6958514925450944) q[1];
rz(-0.6983233527687362) q[1];
ry(1.5010689807906026) q[2];
rz(-2.095036357582405) q[2];
ry(0.5785343284756984) q[3];
rz(-2.4837614242177826) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.2016415391028366) q[0];
rz(2.345812985886189) q[0];
ry(0.4158621682175063) q[1];
rz(0.706162153613378) q[1];
ry(-2.5783060509178233) q[2];
rz(1.12550871410746) q[2];
ry(1.681355211161458) q[3];
rz(2.61919119393358) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(3.0311884388961023) q[0];
rz(-1.769416936372826) q[0];
ry(2.3919909480008616) q[1];
rz(1.0194855231805084) q[1];
ry(2.7769076680696774) q[2];
rz(3.037531783425138) q[2];
ry(-0.3954282083811415) q[3];
rz(2.384672727154696) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.5441607378883564) q[0];
rz(0.2716926001300939) q[0];
ry(-0.2658277046704069) q[1];
rz(1.0887625799718375) q[1];
ry(-0.3648000562789999) q[2];
rz(1.687685866849281) q[2];
ry(-0.5179028508902448) q[3];
rz(0.32824546363368956) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.4628160615082658) q[0];
rz(0.5210588477253877) q[0];
ry(0.019384885341067637) q[1];
rz(-0.08370543171720256) q[1];
ry(0.6532189832941814) q[2];
rz(-2.918280062114068) q[2];
ry(-2.8272058513277156) q[3];
rz(0.7510237745725113) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.2024062584152277) q[0];
rz(1.7717563301314683) q[0];
ry(-0.058275796356801905) q[1];
rz(-1.0586614500834908) q[1];
ry(-2.206987372988562) q[2];
rz(-2.459644748545497) q[2];
ry(2.94233886208935) q[3];
rz(1.3229456878636006) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.9585051658050134) q[0];
rz(-1.1215738719843635) q[0];
ry(2.885843295139577) q[1];
rz(1.036816931191475) q[1];
ry(2.4548245326562497) q[2];
rz(0.9931474361923246) q[2];
ry(2.9319769026951494) q[3];
rz(0.41135143884630043) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.9707888815010187) q[0];
rz(-0.9382681819632248) q[0];
ry(0.5707451070638081) q[1];
rz(-2.2539812368979746) q[1];
ry(-1.608957845436575) q[2];
rz(-2.023791912077274) q[2];
ry(-1.6986115499834955) q[3];
rz(0.6667237266984883) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.17587043042184525) q[0];
rz(-2.688808210842222) q[0];
ry(-2.1773590987128886) q[1];
rz(-0.29880007251399315) q[1];
ry(0.3335154220973715) q[2];
rz(2.1730349507453575) q[2];
ry(1.7972414759797604) q[3];
rz(-0.1258212359382665) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.9701913788128413) q[0];
rz(1.201265693161873) q[0];
ry(2.8556510948893186) q[1];
rz(1.4326500142941485) q[1];
ry(1.0774189747930296) q[2];
rz(-1.5807070500577103) q[2];
ry(-2.2094408329618247) q[3];
rz(-0.15226513644038775) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.7445852415998364) q[0];
rz(-2.7183545369999025) q[0];
ry(-0.2216691421799597) q[1];
rz(0.25966131180321417) q[1];
ry(-0.8865789752803834) q[2];
rz(2.3964808244934197) q[2];
ry(-0.30114154349462824) q[3];
rz(1.9992211061935967) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.045192767556053) q[0];
rz(-2.111775140808069) q[0];
ry(3.103270703414131) q[1];
rz(1.3059664365722172) q[1];
ry(0.944167426375385) q[2];
rz(-1.3523757310051707) q[2];
ry(1.0844637840341633) q[3];
rz(-1.5118821394661168) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.5728106385266583) q[0];
rz(-1.8874614973841737) q[0];
ry(1.9853334522699044) q[1];
rz(1.4611015178446531) q[1];
ry(2.091625737770636) q[2];
rz(2.846686464147461) q[2];
ry(-0.7089885026461047) q[3];
rz(-1.9334540059069694) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.2769850352625651) q[0];
rz(-2.5684620883325464) q[0];
ry(0.008730645807097632) q[1];
rz(2.5402000105146745) q[1];
ry(0.19369145537225144) q[2];
rz(-1.946350726699198) q[2];
ry(0.4738377373941845) q[3];
rz(-0.5474541515945117) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.772350605568383) q[0];
rz(2.0385473472389144) q[0];
ry(-0.7295350062131245) q[1];
rz(-0.5999849918527902) q[1];
ry(0.08571700670557636) q[2];
rz(2.07877402886987) q[2];
ry(2.7853258586024823) q[3];
rz(2.6269647494116195) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.665950428749056) q[0];
rz(1.0885452376992388) q[0];
ry(2.2589440792633946) q[1];
rz(0.07511228014129674) q[1];
ry(-1.515555660760223) q[2];
rz(0.5037084102450509) q[2];
ry(1.5866950501965968) q[3];
rz(0.7926783464071511) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.991132475689525) q[0];
rz(-2.0604491498641218) q[0];
ry(-2.692863099716678) q[1];
rz(0.6796246019621782) q[1];
ry(2.8492772040641565) q[2];
rz(-0.9655883687666817) q[2];
ry(-2.7806575094254473) q[3];
rz(1.1987023611117307) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.278922149044132) q[0];
rz(0.6200220365197677) q[0];
ry(-0.1910134202795273) q[1];
rz(1.8591461698059977) q[1];
ry(1.300614757851803) q[2];
rz(-1.4513235127779478) q[2];
ry(-2.114246471319662) q[3];
rz(-0.2769427883822445) q[3];