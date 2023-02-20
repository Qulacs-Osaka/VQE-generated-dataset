OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(0.005386778023013841) q[0];
rz(3.1248631377033567) q[0];
ry(-0.02755790414039616) q[1];
rz(0.24664496769970334) q[1];
ry(1.580117669978963) q[2];
rz(-1.948650827962879) q[2];
ry(-1.5608517053737172) q[3];
rz(1.834334756619455) q[3];
ry(-0.015308744802899454) q[4];
rz(1.7843994754111185) q[4];
ry(0.001496733740893319) q[5];
rz(1.0376433551271502) q[5];
ry(0.10549620151885102) q[6];
rz(-0.09828912309736103) q[6];
ry(0.1888788713048601) q[7];
rz(0.05884572939114701) q[7];
ry(-3.0665531350130424) q[8];
rz(0.26345132488150946) q[8];
ry(2.8757244281853858) q[9];
rz(2.3472240217551423) q[9];
ry(0.28513779541918277) q[10];
rz(-2.783690418718778) q[10];
ry(-1.481337313776037) q[11];
rz(2.2572096129542243) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(2.0965829173454464) q[0];
rz(0.8558032745446856) q[0];
ry(1.474944937974762) q[1];
rz(1.987310367717825) q[1];
ry(-1.3333716364511599) q[2];
rz(2.808196334772123) q[2];
ry(-1.9522809299160877) q[3];
rz(1.7932557914652918) q[3];
ry(-0.9220407727019901) q[4];
rz(-2.54791412681948) q[4];
ry(2.49346147371872) q[5];
rz(-0.5818006122076662) q[5];
ry(3.0234636799497823) q[6];
rz(-1.3172172567843123) q[6];
ry(-0.1239724368815418) q[7];
rz(-2.052007326891791) q[7];
ry(-3.083508471918207) q[8];
rz(2.6943613155054815) q[8];
ry(0.09288132938838935) q[9];
rz(2.8424208199854215) q[9];
ry(1.4839345114812101) q[10];
rz(1.224002619147588) q[10];
ry(-2.920209472121127) q[11];
rz(0.5264476247831107) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.1571715875630328) q[0];
rz(-2.507581081669316) q[0];
ry(-0.8216240233391564) q[1];
rz(-2.134601166009392) q[1];
ry(2.625823932413126) q[2];
rz(1.072330076259023) q[2];
ry(-2.667375220005624) q[3];
rz(-0.9800200380755226) q[3];
ry(-0.833742686530254) q[4];
rz(2.1035970233784393) q[4];
ry(-0.4164101799704718) q[5];
rz(-0.8888216440658581) q[5];
ry(-1.4275907986483432) q[6];
rz(2.0341631397624624) q[6];
ry(-1.7207830360356315) q[7];
rz(-1.4159375491088104) q[7];
ry(-1.6891115934406873) q[8];
rz(0.8091220853416262) q[8];
ry(0.34167781653561224) q[9];
rz(-1.3122414753981513) q[9];
ry(-1.594886421409283) q[10];
rz(-0.41856396127262124) q[10];
ry(2.295584365941105) q[11];
rz(-1.2750215739130717) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(2.9388529343708405) q[0];
rz(-3.0770783654394545) q[0];
ry(0.25159061798921917) q[1];
rz(2.5160567570779877) q[1];
ry(3.0982455946015386) q[2];
rz(-0.3063410163516451) q[2];
ry(-0.05071382826745996) q[3];
rz(2.4476347475950915) q[3];
ry(0.049318563098492696) q[4];
rz(2.2426672408744652) q[4];
ry(-0.05002986130908216) q[5];
rz(-0.6146026586412034) q[5];
ry(3.1248326313690145) q[6];
rz(-2.0903897735029178) q[6];
ry(-3.124970547278705) q[7];
rz(1.2563553061519765) q[7];
ry(2.1915443203588874) q[8];
rz(2.860470028197463) q[8];
ry(-2.5880221257060954) q[9];
rz(2.0885694677697906) q[9];
ry(2.306832788675346) q[10];
rz(1.2601908650298617) q[10];
ry(2.3794190354971487) q[11];
rz(-0.9599733776914936) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.5363725308884029) q[0];
rz(-1.683661596224644) q[0];
ry(2.5521771435709377) q[1];
rz(-2.1916708149193433) q[1];
ry(2.9508354647494044) q[2];
rz(0.6818517431480223) q[2];
ry(-3.0612836030767774) q[3];
rz(-2.4273744318949744) q[3];
ry(3.112728986552341) q[4];
rz(-0.3872438024536953) q[4];
ry(0.2239064194785191) q[5];
rz(0.18844794222509884) q[5];
ry(-0.04581639113495035) q[6];
rz(0.4159657191769836) q[6];
ry(-0.01848824920672687) q[7];
rz(-0.04701487675573279) q[7];
ry(1.2238475388238808) q[8];
rz(-2.2488079174340823) q[8];
ry(2.019282592975517) q[9];
rz(2.643188452964195) q[9];
ry(-2.028202379240634) q[10];
rz(-0.3616506451253887) q[10];
ry(-0.599145692480718) q[11];
rz(-3.0685606659126905) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.7620740289880559) q[0];
rz(-0.3789703298213701) q[0];
ry(1.598143709490473) q[1];
rz(-1.4532296546157086) q[1];
ry(3.0734752437550403) q[2];
rz(0.9432076653886298) q[2];
ry(-0.04232710704355469) q[3];
rz(3.0570873701011125) q[3];
ry(0.09872102274797616) q[4];
rz(-1.3924302721930546) q[4];
ry(0.011721294992121045) q[5];
rz(-0.5139601170415311) q[5];
ry(-1.528019965810372) q[6];
rz(1.615001867765508) q[6];
ry(1.6049735271304133) q[7];
rz(1.53370062847419) q[7];
ry(-1.26464726287754) q[8];
rz(2.0052209407062844) q[8];
ry(2.498190308035531) q[9];
rz(1.7544632072340782) q[9];
ry(-3.0815980591498433) q[10];
rz(-0.35183954043057314) q[10];
ry(-3.0560741100684625) q[11];
rz(-1.7639265132140027) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.8774560065585888) q[0];
rz(-1.1553933132656962) q[0];
ry(3.066424599404895) q[1];
rz(-2.71538954116399) q[1];
ry(-3.0919249646201146) q[2];
rz(-0.7802969898758922) q[2];
ry(-3.090536956650149) q[3];
rz(1.961439876904674) q[3];
ry(-0.35883127153129335) q[4];
rz(1.2098173242144776) q[4];
ry(2.6307125881438185) q[5];
rz(1.1269928214352856) q[5];
ry(1.4459820821539917) q[6];
rz(1.7273250546393681) q[6];
ry(1.645220113932953) q[7];
rz(-0.2523244685082422) q[7];
ry(-1.6524217556013463) q[8];
rz(-2.3854247936561697) q[8];
ry(2.9865933266371107) q[9];
rz(-2.7174801579508907) q[9];
ry(0.46733632453722596) q[10];
rz(-1.3614077321490468) q[10];
ry(0.2227734146493701) q[11];
rz(2.108992276100752) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-0.3262252916311213) q[0];
rz(-2.1410160106143925) q[0];
ry(-2.4882097510882577) q[1];
rz(1.341079841631175) q[1];
ry(-3.05595532060515) q[2];
rz(1.4891902735392097) q[2];
ry(-0.0999521598775472) q[3];
rz(-2.165027694101204) q[3];
ry(-1.0653650943328472) q[4];
rz(-1.844396417344648) q[4];
ry(0.676806053863713) q[5];
rz(-0.558164174799317) q[5];
ry(-0.15994997634999564) q[6];
rz(-1.7104441665328007) q[6];
ry(3.108243675960676) q[7];
rz(2.8844300238353973) q[7];
ry(2.457307330436811) q[8];
rz(-1.9898735348930587) q[8];
ry(1.7126428300066143) q[9];
rz(1.8945097806483011) q[9];
ry(-1.7229391388576658) q[10];
rz(-0.04728708574616912) q[10];
ry(-1.6083959692578853) q[11];
rz(1.3216146602586596) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(2.2774985299584185) q[0];
rz(-2.5710039604696404) q[0];
ry(2.413625924070194) q[1];
rz(1.2556936401952912) q[1];
ry(1.654171510808564) q[2];
rz(-0.7144994241094997) q[2];
ry(-1.2930310299651877) q[3];
rz(-0.12925766821238127) q[3];
ry(0.25710634730634224) q[4];
rz(1.2656103562154046) q[4];
ry(0.8800444611022771) q[5];
rz(3.009119048138474) q[5];
ry(1.5723326074262243) q[6];
rz(-1.5764685495232138) q[6];
ry(-1.5651298976395251) q[7];
rz(1.5779061425491099) q[7];
ry(0.030591369014699055) q[8];
rz(1.2032207381457867) q[8];
ry(-3.1314737851891192) q[9];
rz(2.560457022741258) q[9];
ry(-1.6601499077919808) q[10];
rz(1.9021747490382515) q[10];
ry(0.5493138799293086) q[11];
rz(2.3161239301961976) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-2.988807709867047) q[0];
rz(-1.6778283853210596) q[0];
ry(-0.04559216430528251) q[1];
rz(-2.041504645187925) q[1];
ry(-0.01588194870916701) q[2];
rz(0.26479639213282097) q[2];
ry(0.010151409544552514) q[3];
rz(-1.93284327054522) q[3];
ry(2.6909895771791734) q[4];
rz(-0.8587691767599318) q[4];
ry(-1.4826128704689108) q[5];
rz(0.854307711749879) q[5];
ry(-2.6752125908966535) q[6];
rz(1.565449974907601) q[6];
ry(-0.5187615794732453) q[7];
rz(1.559461273033369) q[7];
ry(-3.0405324890874947) q[8];
rz(-1.237368263403468) q[8];
ry(1.3406031345544474) q[9];
rz(2.954664215191522) q[9];
ry(-1.6154915217034083) q[10];
rz(2.550377211179313) q[10];
ry(0.12729656139743453) q[11];
rz(-1.4061097986368474) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.2699243620115874) q[0];
rz(-2.199163407765223) q[0];
ry(2.5557176106256607) q[1];
rz(0.8870464409133456) q[1];
ry(0.010448817012644442) q[2];
rz(-3.103219168439204) q[2];
ry(1.9797151750998718) q[3];
rz(-1.5606561736627766) q[3];
ry(0.9008075167043295) q[4];
rz(-0.4052778332358715) q[4];
ry(-0.056039602696557715) q[5];
rz(-0.5862884964746549) q[5];
ry(-1.5714823080469476) q[6];
rz(0.21430104957429205) q[6];
ry(1.5702050326987986) q[7];
rz(3.1378929166392324) q[7];
ry(3.1217938030809513) q[8];
rz(-1.7096389189179373) q[8];
ry(3.0807508343063073) q[9];
rz(2.172366978097579) q[9];
ry(2.919794797299356) q[10];
rz(-1.0043534919263346) q[10];
ry(3.122531335573884) q[11];
rz(0.231838224708258) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.8798508309463209) q[0];
rz(1.290480376331412) q[0];
ry(2.7320393891761072) q[1];
rz(-1.1403723502603453) q[1];
ry(3.1032205202614733) q[2];
rz(0.5895798884973353) q[2];
ry(0.8416932450419001) q[3];
rz(1.5860623131962652) q[3];
ry(-2.856149711881108) q[4];
rz(-0.5943841099570567) q[4];
ry(0.17679123425521523) q[5];
rz(-1.7099139380138553) q[5];
ry(1.5026470518365136) q[6];
rz(0.8835479404419972) q[6];
ry(-1.5952181234890055) q[7];
rz(2.8759079859079653) q[7];
ry(1.137771832536212) q[8];
rz(0.5588176861309488) q[8];
ry(2.370203871972368) q[9];
rz(-0.15766838816186546) q[9];
ry(-0.9510734488712291) q[10];
rz(2.5971959966025504) q[10];
ry(-1.6766057085217674) q[11];
rz(0.1517546600415427) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.0406737300956032) q[0];
rz(1.1221505375876903) q[0];
ry(-3.1332607001932606) q[1];
rz(-1.3607065229122917) q[1];
ry(-1.984139384615304) q[2];
rz(-2.7175924609307587) q[2];
ry(1.9859530361892226) q[3];
rz(1.6109671394132616) q[3];
ry(-2.3217261829315725) q[4];
rz(0.9276526522542907) q[4];
ry(-0.8027311881601173) q[5];
rz(0.2249046754337305) q[5];
ry(-0.0014633364257840985) q[6];
rz(1.5056698581985204) q[6];
ry(-3.139509054449945) q[7];
rz(-0.8585962309448273) q[7];
ry(-0.021504341009102035) q[8];
rz(-0.6176631416421402) q[8];
ry(-3.1284701081749398) q[9];
rz(-0.5249055142981236) q[9];
ry(1.6113075424812457) q[10];
rz(0.33033091343976023) q[10];
ry(0.3381486692439317) q[11];
rz(1.1293071255928613) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.8937158149305489) q[0];
rz(-2.0012850978442183) q[0];
ry(-3.0270239799843606) q[1];
rz(2.7403120077905907) q[1];
ry(-0.00014068526496754) q[2];
rz(2.7461765636661846) q[2];
ry(0.11725279593946514) q[3];
rz(-1.5976124063311463) q[3];
ry(-2.9875374557314447) q[4];
rz(0.4097044565029466) q[4];
ry(1.1992185965939273) q[5];
rz(0.41571821898499556) q[5];
ry(1.0178665174692791) q[6];
rz(0.2604388096590351) q[6];
ry(-1.49198817975554) q[7];
rz(2.758147195173756) q[7];
ry(-2.6689057760915094) q[8];
rz(1.965065071058618) q[8];
ry(0.7198160659112771) q[9];
rz(2.3725435889039015) q[9];
ry(-0.2814521087855193) q[10];
rz(-0.6924910892825134) q[10];
ry(-2.1404847309392094) q[11];
rz(1.1287216739458095) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-3.1389710859392355) q[0];
rz(-1.3871733092675669) q[0];
ry(3.136680218236818) q[1];
rz(1.068323644047351) q[1];
ry(-2.317324377800958) q[2];
rz(-1.5643078451271912) q[2];
ry(-0.8728165661092682) q[3];
rz(1.0385642213983175) q[3];
ry(1.5235869018964088) q[4];
rz(-3.0633706528964337) q[4];
ry(-1.2089301145879645) q[5];
rz(3.134555368869034) q[5];
ry(3.1412666449830513) q[6];
rz(0.36345082809246865) q[6];
ry(3.137477017530513) q[7];
rz(1.5615823552624488) q[7];
ry(-3.132066589442191) q[8];
rz(-2.010088762012357) q[8];
ry(0.04919431722601609) q[9];
rz(1.1860010188838528) q[9];
ry(2.411040003276692) q[10];
rz(1.4078323986390413) q[10];
ry(2.8678670406155438) q[11];
rz(-2.6584596355478753) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-3.078658915399582) q[0];
rz(-2.4506215995803466) q[0];
ry(0.13473798401528855) q[1];
rz(2.9002156829256416) q[1];
ry(1.9818894048086968) q[2];
rz(1.6092718732365805) q[2];
ry(3.1097164020498242) q[3];
rz(-2.0759506076849075) q[3];
ry(1.4252457701528605) q[4];
rz(-1.019688378475247) q[4];
ry(-1.5519372693139972) q[5];
rz(2.5379208960088984) q[5];
ry(1.5335953785276935) q[6];
rz(-1.7546231170148445) q[6];
ry(1.7682908143889087) q[7];
rz(2.9136902178213964) q[7];
ry(-1.5623101748458534) q[8];
rz(1.1946694232084785) q[8];
ry(-0.6819820033070885) q[9];
rz(-0.039638025369838736) q[9];
ry(-0.45037018849295185) q[10];
rz(-2.555624736935204) q[10];
ry(2.8343419777992067) q[11];
rz(2.1922147896068154) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.2469686208861344) q[0];
rz(-1.1938315076554458) q[0];
ry(1.0876363273752014) q[1];
rz(0.0369988847091669) q[1];
ry(-2.594958190045548) q[2];
rz(2.747045953390318) q[2];
ry(1.3783991203279256) q[3];
rz(0.43083559564057555) q[3];
ry(-1.3942619366860836) q[4];
rz(-2.757872962715759) q[4];
ry(1.4231564223081268) q[5];
rz(-2.2323680998523683) q[5];
ry(0.004457733518322904) q[6];
rz(0.13901678204457618) q[6];
ry(-0.02273000128737438) q[7];
rz(-1.3200198872319036) q[7];
ry(3.1415110694930677) q[8];
rz(2.8399503581633283) q[8];
ry(0.007857533920044648) q[9];
rz(-0.044966827460632075) q[9];
ry(-2.0857072747770076) q[10];
rz(1.3451632812267942) q[10];
ry(1.7953746157591264) q[11];
rz(-0.14976476671726346) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.8250277385368237) q[0];
rz(3.087681618535113) q[0];
ry(1.471239225480534) q[1];
rz(-2.3408917843850072) q[1];
ry(-0.019689448896532017) q[2];
rz(-1.1370363080913448) q[2];
ry(-0.00546803517445823) q[3];
rz(-2.019211167840048) q[3];
ry(0.0024924861951047816) q[4];
rz(2.482650216565627) q[4];
ry(-0.001895506919510126) q[5];
rz(1.9415973460600702) q[5];
ry(1.595752954831537) q[6];
rz(1.0299771574948877) q[6];
ry(-1.5029907867660859) q[7];
rz(-1.025321596516461) q[7];
ry(-3.1266810722700273) q[8];
rz(-2.24937134217097) q[8];
ry(-0.05480470508127145) q[9];
rz(2.8737368802430523) q[9];
ry(-2.8839655196720795) q[10];
rz(1.0127098177437275) q[10];
ry(-2.786319733141813) q[11];
rz(-2.703052461369411) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.486517071904288) q[0];
rz(1.9835452745748257) q[0];
ry(-1.7745933245941723) q[1];
rz(-0.8783783228154476) q[1];
ry(-1.5617454860416964) q[2];
rz(1.0761638961416633) q[2];
ry(1.5929297766496968) q[3];
rz(-0.5393925237241658) q[3];
ry(-0.7114556285842434) q[4];
rz(1.977747267348678) q[4];
ry(0.11042082387352789) q[5];
rz(-1.0610835911419967) q[5];
ry(3.1191197627152847) q[6];
rz(1.393188787638108) q[6];
ry(-3.1394560086931733) q[7];
rz(1.9546899430174716) q[7];
ry(-1.623085409709649) q[8];
rz(3.14020931362672) q[8];
ry(1.5755104843465322) q[9];
rz(-0.09573217484515821) q[9];
ry(2.307251435648783) q[10];
rz(1.5948034014683856) q[10];
ry(-1.2570364725669227) q[11];
rz(0.7915320965832632) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(2.9709122311014466) q[0];
rz(0.22501137492648232) q[0];
ry(-1.6358499234110084) q[1];
rz(1.641555851701268) q[1];
ry(-0.0007285004716132917) q[2];
rz(2.257124550028314) q[2];
ry(3.131251254045361) q[3];
rz(-0.3922133897213165) q[3];
ry(1.6293567337195887) q[4];
rz(-1.8008120725661974) q[4];
ry(1.4043530816176097) q[5];
rz(-1.7100456425081683) q[5];
ry(-3.1405963626079565) q[6];
rz(1.8092150924586292) q[6];
ry(-3.1414917153686837) q[7];
rz(-1.625746415325189) q[7];
ry(-1.5680522409372353) q[8];
rz(-2.0486817604171224) q[8];
ry(2.6294878114308893) q[9];
rz(0.4037698282494224) q[9];
ry(-3.087919933733129) q[10];
rz(-2.354514103836419) q[10];
ry(2.0857223612272593) q[11];
rz(-1.0454439324256675) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-3.01411148873469) q[0];
rz(1.0267345711900546) q[0];
ry(-1.5129391188442678) q[1];
rz(0.7414329370221246) q[1];
ry(-1.5506517884815396) q[2];
rz(2.487691563172168) q[2];
ry(-1.5723574697043579) q[3];
rz(-3.070948463373464) q[3];
ry(-1.5308634512951538) q[4];
rz(-0.26773188543677373) q[4];
ry(-0.9343705936773522) q[5];
rz(0.499772964515979) q[5];
ry(-1.5245452412667173) q[6];
rz(0.058052919596333644) q[6];
ry(1.579363979484581) q[7];
rz(2.316424842164304) q[7];
ry(0.012065964744607527) q[8];
rz(-0.9821891609967237) q[8];
ry(0.19698864712241537) q[9];
rz(0.33188707565201864) q[9];
ry(1.577011055542833) q[10];
rz(2.0418027031739583) q[10];
ry(-3.079834351392503) q[11];
rz(-1.7832335902842837) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.920127471450438) q[0];
rz(0.9324214194784313) q[0];
ry(-1.9421871489990874) q[1];
rz(-0.1887898459189898) q[1];
ry(-3.1232642273972266) q[2];
rz(-2.1325717211696675) q[2];
ry(-0.38506524028721656) q[3];
rz(1.5083043717844908) q[3];
ry(1.590423852526883) q[4];
rz(-3.133268595484282) q[4];
ry(-1.5707209331696597) q[5];
rz(0.00524386424828469) q[5];
ry(-3.1374199794318143) q[6];
rz(-0.692871442124247) q[6];
ry(-0.01243797590384782) q[7];
rz(1.0007722384906437) q[7];
ry(3.1348536273534684) q[8];
rz(-1.8014020469623027) q[8];
ry(-0.0020512798514635833) q[9];
rz(-2.3911526759141344) q[9];
ry(-0.07129058902597496) q[10];
rz(2.608895793916655) q[10];
ry(1.5358403640337772) q[11];
rz(-0.11636698944151558) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.04018070436375076) q[0];
rz(-0.727452834339566) q[0];
ry(-1.463696823865463) q[1];
rz(1.0113851710889241) q[1];
ry(-0.699495003517564) q[2];
rz(2.7369463085245553) q[2];
ry(-2.4281944374537257) q[3];
rz(0.3354579403956313) q[3];
ry(1.5651328083038958) q[4];
rz(3.0819730412051696) q[4];
ry(-1.569486397472894) q[5];
rz(-3.0767544844615218) q[5];
ry(-0.05486581763284004) q[6];
rz(2.7872956100397075) q[6];
ry(0.4148201499626317) q[7];
rz(0.017330999023089433) q[7];
ry(-0.16058441516759459) q[8];
rz(-1.048602107582414) q[8];
ry(-1.8587907029452049) q[9];
rz(0.8835009955547469) q[9];
ry(-0.053190482630238727) q[10];
rz(0.0572543869817137) q[10];
ry(-3.0918442264627686) q[11];
rz(-0.08859991057385663) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-0.013064335698510732) q[0];
rz(-1.1977225056091259) q[0];
ry(-3.0886225944929393) q[1];
rz(-2.1326920557180156) q[1];
ry(-0.3557851497030991) q[2];
rz(-0.4631042501460607) q[2];
ry(-1.5511937835995173) q[3];
rz(2.7398398521902196) q[3];
ry(-1.6042963383399835) q[4];
rz(-0.3140866214561004) q[4];
ry(-1.5826231451863355) q[5];
rz(1.5640803762020012) q[5];
ry(-1.565054374344597) q[6];
rz(0.000801046080271087) q[6];
ry(1.5737204711557347) q[7];
rz(3.1407439433596616) q[7];
ry(-0.03556284811046506) q[8];
rz(-1.563737345543611) q[8];
ry(0.0074955277571762124) q[9];
rz(-0.8892275141222733) q[9];
ry(-1.5480789396237407) q[10];
rz(-0.3455488881866628) q[10];
ry(-1.6653796343784075) q[11];
rz(-2.163431243854503) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-3.1005874608527346) q[0];
rz(-0.031206037744774484) q[0];
ry(-1.5808650069076766) q[1];
rz(1.5959382130067747) q[1];
ry(3.111771787506625) q[2];
rz(-2.3725554215788596) q[2];
ry(-0.011155085339108755) q[3];
rz(0.4816115746214816) q[3];
ry(-3.137551080518032) q[4];
rz(-2.4068129707958783) q[4];
ry(-3.1367961348226348) q[5];
rz(-0.30688395041551075) q[5];
ry(1.5646038403765745) q[6];
rz(2.6410955888527647) q[6];
ry(-1.567710442648492) q[7];
rz(-2.666368400793269) q[7];
ry(-0.009279431221850654) q[8];
rz(1.9564841637140309) q[8];
ry(2.8253930397810336) q[9];
rz(1.5732749730840445) q[9];
ry(0.8576339261736213) q[10];
rz(-1.6444296298934042) q[10];
ry(0.4032880358610526) q[11];
rz(-2.8687952252519) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-0.012347523295794402) q[0];
rz(2.1102753847140026) q[0];
ry(-1.521400724955475) q[1];
rz(-1.4191147912082076) q[1];
ry(1.5410466108595262) q[2];
rz(2.0679117641613347) q[2];
ry(-1.5733321656090349) q[3];
rz(2.0613183819984826) q[3];
ry(1.1291763140049316) q[4];
rz(1.7607206654062966) q[4];
ry(1.2718393815028468) q[5];
rz(0.017257079512623896) q[5];
ry(-0.28333916258729896) q[6];
rz(-0.6523825610786993) q[6];
ry(-1.837539653889858) q[7];
rz(-0.7345742702437591) q[7];
ry(0.09941920525768655) q[8];
rz(-2.163452815829076) q[8];
ry(-1.5300907235158867) q[9];
rz(-1.1583859357158444) q[9];
ry(-0.009437489023663481) q[10];
rz(2.280190509104026) q[10];
ry(3.1040192807608338) q[11];
rz(1.129195017566925) q[11];