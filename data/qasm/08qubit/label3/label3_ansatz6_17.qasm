OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(1.9158824246086696) q[0];
ry(0.8471337916760528) q[1];
cx q[0],q[1];
ry(2.341784509103149) q[0];
ry(0.07799012635507228) q[1];
cx q[0],q[1];
ry(0.8150113453012047) q[1];
ry(-0.7838267533187475) q[2];
cx q[1],q[2];
ry(-1.5885766147725646) q[1];
ry(2.9466493474439512) q[2];
cx q[1],q[2];
ry(1.2709641835625658) q[2];
ry(2.6999175412890537) q[3];
cx q[2],q[3];
ry(-2.3148996876548606) q[2];
ry(-0.8485024562611727) q[3];
cx q[2],q[3];
ry(0.05130679382468628) q[3];
ry(-0.6380601670778865) q[4];
cx q[3],q[4];
ry(-0.18072608083938654) q[3];
ry(1.2585302094008919) q[4];
cx q[3],q[4];
ry(-1.3779464240252093) q[4];
ry(-0.9191523577755643) q[5];
cx q[4],q[5];
ry(-1.2398638108301387) q[4];
ry(1.2687125225149147) q[5];
cx q[4],q[5];
ry(0.7685360002512267) q[5];
ry(1.4889228354183999) q[6];
cx q[5],q[6];
ry(-1.748226184586796) q[5];
ry(1.580043563681481) q[6];
cx q[5],q[6];
ry(-0.9828972265320921) q[6];
ry(-0.8597723782812157) q[7];
cx q[6],q[7];
ry(-2.9842443412303314) q[6];
ry(2.5288932629535226) q[7];
cx q[6],q[7];
ry(0.6395831146856006) q[0];
ry(1.1817426037271543) q[1];
cx q[0],q[1];
ry(-1.5638847643729887) q[0];
ry(1.3613980586727426) q[1];
cx q[0],q[1];
ry(-2.914700169210219) q[1];
ry(-0.0898928260192049) q[2];
cx q[1],q[2];
ry(1.5638396622867443) q[1];
ry(2.769449455088563) q[2];
cx q[1],q[2];
ry(2.633538875744066) q[2];
ry(-2.3329309210857243) q[3];
cx q[2],q[3];
ry(2.3460619399420146) q[2];
ry(-1.4685604713910252) q[3];
cx q[2],q[3];
ry(-3.139705464176415) q[3];
ry(1.058280083081772) q[4];
cx q[3],q[4];
ry(-1.1404714296783922) q[3];
ry(1.9605360450569957) q[4];
cx q[3],q[4];
ry(1.7630589768010494) q[4];
ry(-3.068188148059931) q[5];
cx q[4],q[5];
ry(-2.8498235389837214) q[4];
ry(-1.0118459014364847) q[5];
cx q[4],q[5];
ry(1.6140407344355545) q[5];
ry(-2.448028303702897) q[6];
cx q[5],q[6];
ry(0.35954819218959067) q[5];
ry(2.4530341602373094) q[6];
cx q[5],q[6];
ry(2.08286502259313) q[6];
ry(1.301389932209256) q[7];
cx q[6],q[7];
ry(1.809645475179533) q[6];
ry(-2.207497043615227) q[7];
cx q[6],q[7];
ry(1.1855922215361812) q[0];
ry(-0.33911718668762436) q[1];
cx q[0],q[1];
ry(2.5393668341019757) q[0];
ry(-1.6206642546524173) q[1];
cx q[0],q[1];
ry(-2.9585472511398847) q[1];
ry(1.436525251329732) q[2];
cx q[1],q[2];
ry(2.8419056265715787) q[1];
ry(1.542619810991827) q[2];
cx q[1],q[2];
ry(-0.26851410992276653) q[2];
ry(1.0961918001138427) q[3];
cx q[2],q[3];
ry(1.6058676778693486) q[2];
ry(-1.298674996904891) q[3];
cx q[2],q[3];
ry(1.7733388219062187) q[3];
ry(-2.42460434273894) q[4];
cx q[3],q[4];
ry(-1.8032217700747322) q[3];
ry(0.31627407359774506) q[4];
cx q[3],q[4];
ry(-2.5565886518841605) q[4];
ry(0.9453159385692596) q[5];
cx q[4],q[5];
ry(2.068277180613871) q[4];
ry(1.3492940207010096) q[5];
cx q[4],q[5];
ry(2.572912192874994) q[5];
ry(-0.30844549829991014) q[6];
cx q[5],q[6];
ry(0.31954526922060056) q[5];
ry(0.517797055000443) q[6];
cx q[5],q[6];
ry(-2.3488002266359036) q[6];
ry(-2.538730662775051) q[7];
cx q[6],q[7];
ry(-0.5925374473441858) q[6];
ry(3.107845257073365) q[7];
cx q[6],q[7];
ry(-3.0984902024788226) q[0];
ry(-3.0809583481844207) q[1];
cx q[0],q[1];
ry(1.3663930089516816) q[0];
ry(-0.7189566723874661) q[1];
cx q[0],q[1];
ry(-2.7134006628801943) q[1];
ry(-1.979969488652673) q[2];
cx q[1],q[2];
ry(0.9808865502642865) q[1];
ry(0.09097826565872147) q[2];
cx q[1],q[2];
ry(0.89767223549471) q[2];
ry(0.39466588249726176) q[3];
cx q[2],q[3];
ry(1.0086624892865839) q[2];
ry(-0.45271690165039136) q[3];
cx q[2],q[3];
ry(2.8881290829165334) q[3];
ry(-0.3611431077543109) q[4];
cx q[3],q[4];
ry(-1.3581593698171412) q[3];
ry(0.9918651265629925) q[4];
cx q[3],q[4];
ry(-1.6257597213684472) q[4];
ry(-0.5926317791093823) q[5];
cx q[4],q[5];
ry(-1.72204191666937) q[4];
ry(1.9707424639014404) q[5];
cx q[4],q[5];
ry(-1.9891611135914753) q[5];
ry(-1.3946625971045319) q[6];
cx q[5],q[6];
ry(-2.7789906238888205) q[5];
ry(-2.464630442876337) q[6];
cx q[5],q[6];
ry(-2.338109546791279) q[6];
ry(0.5020761036942895) q[7];
cx q[6],q[7];
ry(-2.548711472510781) q[6];
ry(-2.2623089887575603) q[7];
cx q[6],q[7];
ry(0.03564735782384343) q[0];
ry(3.1217674403861677) q[1];
cx q[0],q[1];
ry(-0.12037730282911262) q[0];
ry(-1.1918304959261912) q[1];
cx q[0],q[1];
ry(-3.096953369092728) q[1];
ry(2.481088898793525) q[2];
cx q[1],q[2];
ry(1.119210179043277) q[1];
ry(-1.9327182170091302) q[2];
cx q[1],q[2];
ry(-2.115061192911136) q[2];
ry(-0.8208988200803367) q[3];
cx q[2],q[3];
ry(-2.5312957903516735) q[2];
ry(-0.5662497296308531) q[3];
cx q[2],q[3];
ry(2.8117941853195205) q[3];
ry(2.051322812055176) q[4];
cx q[3],q[4];
ry(3.0819287640392314) q[3];
ry(0.01604838816924925) q[4];
cx q[3],q[4];
ry(-1.9844714123114144) q[4];
ry(0.4360224114040969) q[5];
cx q[4],q[5];
ry(-0.4334315246744704) q[4];
ry(0.47518388655629273) q[5];
cx q[4],q[5];
ry(2.33620667189403) q[5];
ry(1.4065090447532205) q[6];
cx q[5],q[6];
ry(-0.250039221182247) q[5];
ry(-3.01080336673454) q[6];
cx q[5],q[6];
ry(-2.4390116775016555) q[6];
ry(1.7583406381312046) q[7];
cx q[6],q[7];
ry(-0.553726021348936) q[6];
ry(-0.9958760755954066) q[7];
cx q[6],q[7];
ry(0.9247711858312445) q[0];
ry(0.6868374547612008) q[1];
cx q[0],q[1];
ry(-1.3220590870968216) q[0];
ry(-0.6155460072883965) q[1];
cx q[0],q[1];
ry(2.628551987022889) q[1];
ry(0.8894989285919329) q[2];
cx q[1],q[2];
ry(0.2538242545164055) q[1];
ry(0.7471805115993897) q[2];
cx q[1],q[2];
ry(-2.205651510201283) q[2];
ry(1.8098408065603193) q[3];
cx q[2],q[3];
ry(0.3729803265018363) q[2];
ry(1.501933535688421) q[3];
cx q[2],q[3];
ry(0.9607065733830513) q[3];
ry(0.5047617083558619) q[4];
cx q[3],q[4];
ry(-2.880804461716374) q[3];
ry(0.381874201974322) q[4];
cx q[3],q[4];
ry(-1.6390925913993533) q[4];
ry(1.0245011281757295) q[5];
cx q[4],q[5];
ry(1.6534950909869277) q[4];
ry(-0.1460285184803003) q[5];
cx q[4],q[5];
ry(-0.9358036277906703) q[5];
ry(-1.018889940849529) q[6];
cx q[5],q[6];
ry(-1.156223458156469) q[5];
ry(0.40136579347406054) q[6];
cx q[5],q[6];
ry(-2.0286962688761774) q[6];
ry(-2.589326944106236) q[7];
cx q[6],q[7];
ry(1.0127673903518728) q[6];
ry(-1.183342862134641) q[7];
cx q[6],q[7];
ry(1.3053864774251347) q[0];
ry(-0.9587290017325483) q[1];
cx q[0],q[1];
ry(-1.0315266208868827) q[0];
ry(1.24494909860048) q[1];
cx q[0],q[1];
ry(0.0026730888424975063) q[1];
ry(-2.91765768433726) q[2];
cx q[1],q[2];
ry(-0.6951845500994762) q[1];
ry(-1.8418982290882964) q[2];
cx q[1],q[2];
ry(1.1944244130029142) q[2];
ry(2.4175444397863166) q[3];
cx q[2],q[3];
ry(0.8179991849116979) q[2];
ry(3.0980823085523714) q[3];
cx q[2],q[3];
ry(0.09713658023060534) q[3];
ry(-3.003297488918948) q[4];
cx q[3],q[4];
ry(-0.9933299811442611) q[3];
ry(-0.9536168832687029) q[4];
cx q[3],q[4];
ry(1.169130228732251) q[4];
ry(1.829130977914474) q[5];
cx q[4],q[5];
ry(1.3229482882491495) q[4];
ry(-1.7123013291490221) q[5];
cx q[4],q[5];
ry(-1.59398651386075) q[5];
ry(1.3378293182522951) q[6];
cx q[5],q[6];
ry(0.2969834004262033) q[5];
ry(-2.782473004770546) q[6];
cx q[5],q[6];
ry(2.520170162545788) q[6];
ry(3.138230889736347) q[7];
cx q[6],q[7];
ry(2.0531628696970645) q[6];
ry(2.729842845939145) q[7];
cx q[6],q[7];
ry(-0.778003332121483) q[0];
ry(0.18755393348683125) q[1];
cx q[0],q[1];
ry(-2.4344026989448984) q[0];
ry(2.7878312051172722) q[1];
cx q[0],q[1];
ry(3.017121702135834) q[1];
ry(3.1035965731560977) q[2];
cx q[1],q[2];
ry(-0.9440841331472403) q[1];
ry(-0.11892932648272697) q[2];
cx q[1],q[2];
ry(1.5743428703575617) q[2];
ry(-2.4319828400266985) q[3];
cx q[2],q[3];
ry(-2.209797918413692) q[2];
ry(-0.808562763593864) q[3];
cx q[2],q[3];
ry(1.445137237857998) q[3];
ry(1.0332435995069529) q[4];
cx q[3],q[4];
ry(-0.404253795395982) q[3];
ry(-1.9798602761502533) q[4];
cx q[3],q[4];
ry(1.2318617906912495) q[4];
ry(-1.3851858496192038) q[5];
cx q[4],q[5];
ry(-2.6197136379500088) q[4];
ry(1.6536513032039766) q[5];
cx q[4],q[5];
ry(0.39373700295503106) q[5];
ry(-0.5434074174610571) q[6];
cx q[5],q[6];
ry(1.3253299927183715) q[5];
ry(1.353959541678459) q[6];
cx q[5],q[6];
ry(-1.1160047243959623) q[6];
ry(-2.9216684664743044) q[7];
cx q[6],q[7];
ry(0.524760611931729) q[6];
ry(0.49681094533547226) q[7];
cx q[6],q[7];
ry(-1.377839107467402) q[0];
ry(-3.035540165727256) q[1];
cx q[0],q[1];
ry(-2.6599896824836433) q[0];
ry(-2.1189106424361395) q[1];
cx q[0],q[1];
ry(-1.3031403196029756) q[1];
ry(1.2404365939142254) q[2];
cx q[1],q[2];
ry(-1.3233929817814918) q[1];
ry(-1.0762498870727812) q[2];
cx q[1],q[2];
ry(0.5451560165513809) q[2];
ry(-0.8983188374384968) q[3];
cx q[2],q[3];
ry(2.5285237180725715) q[2];
ry(1.3752413412318882) q[3];
cx q[2],q[3];
ry(-0.0018470660742906375) q[3];
ry(0.2243103332554277) q[4];
cx q[3],q[4];
ry(-1.844059094349772) q[3];
ry(-1.9789449576537894) q[4];
cx q[3],q[4];
ry(1.761622493261716) q[4];
ry(1.2624999278451154) q[5];
cx q[4],q[5];
ry(1.9202125947282265) q[4];
ry(-0.6793184835184164) q[5];
cx q[4],q[5];
ry(-2.4273658681590478) q[5];
ry(-0.9804823396650688) q[6];
cx q[5],q[6];
ry(1.2961612453658817) q[5];
ry(2.519869147136631) q[6];
cx q[5],q[6];
ry(0.32201704321915237) q[6];
ry(-1.9747060921431911) q[7];
cx q[6],q[7];
ry(1.6945104259314148) q[6];
ry(2.907226814192063) q[7];
cx q[6],q[7];
ry(0.27454691531627157) q[0];
ry(1.6977392802290465) q[1];
cx q[0],q[1];
ry(0.46317815439654725) q[0];
ry(-0.7066148845419841) q[1];
cx q[0],q[1];
ry(1.8278265787263415) q[1];
ry(1.5733788219184417) q[2];
cx q[1],q[2];
ry(-1.6156823957572977) q[1];
ry(2.569282998651353) q[2];
cx q[1],q[2];
ry(0.7949830925844381) q[2];
ry(1.173730990011875) q[3];
cx q[2],q[3];
ry(2.7127040413967176) q[2];
ry(-1.1456209746304538) q[3];
cx q[2],q[3];
ry(0.06756740786402027) q[3];
ry(-0.1701090645418155) q[4];
cx q[3],q[4];
ry(-0.561229479759742) q[3];
ry(-0.3336627826894647) q[4];
cx q[3],q[4];
ry(1.1817506256680987) q[4];
ry(1.45469451970772) q[5];
cx q[4],q[5];
ry(1.1391962268783444) q[4];
ry(-2.436892772165887) q[5];
cx q[4],q[5];
ry(0.28745515596346755) q[5];
ry(0.6227256709646237) q[6];
cx q[5],q[6];
ry(-2.6446589775517992) q[5];
ry(0.8073745819920743) q[6];
cx q[5],q[6];
ry(-2.43667336195278) q[6];
ry(2.413705437219443) q[7];
cx q[6],q[7];
ry(-0.6695805280050463) q[6];
ry(-2.4303821058733797) q[7];
cx q[6],q[7];
ry(-0.12438039205086926) q[0];
ry(-2.4877282142860104) q[1];
cx q[0],q[1];
ry(3.105337125893396) q[0];
ry(2.4280829235822114) q[1];
cx q[0],q[1];
ry(2.634652803570923) q[1];
ry(2.476910912982756) q[2];
cx q[1],q[2];
ry(0.017102374390619346) q[1];
ry(2.5641753669975462) q[2];
cx q[1],q[2];
ry(-1.7555380853743712) q[2];
ry(0.8376973075631244) q[3];
cx q[2],q[3];
ry(-2.284762833760221) q[2];
ry(-2.873323138970893) q[3];
cx q[2],q[3];
ry(2.499375692873734) q[3];
ry(1.5034003219385583) q[4];
cx q[3],q[4];
ry(0.9823859489747662) q[3];
ry(-2.913536801067673) q[4];
cx q[3],q[4];
ry(0.829641776965667) q[4];
ry(1.9543851920835529) q[5];
cx q[4],q[5];
ry(-0.47777123039333347) q[4];
ry(-0.3174942913377823) q[5];
cx q[4],q[5];
ry(-0.43247102034035834) q[5];
ry(0.36517751144601807) q[6];
cx q[5],q[6];
ry(-0.6301181853699003) q[5];
ry(-1.2648188562678262) q[6];
cx q[5],q[6];
ry(-2.1318191282479697) q[6];
ry(1.4716957357151594) q[7];
cx q[6],q[7];
ry(-1.2601943797888437) q[6];
ry(2.6117781273847513) q[7];
cx q[6],q[7];
ry(1.9478506091002634) q[0];
ry(-0.18240555301901562) q[1];
cx q[0],q[1];
ry(0.9081800419276066) q[0];
ry(1.7966906062236143) q[1];
cx q[0],q[1];
ry(0.4859549736874634) q[1];
ry(-0.10787448871092842) q[2];
cx q[1],q[2];
ry(2.671746413902877) q[1];
ry(-1.0232108247583858) q[2];
cx q[1],q[2];
ry(1.7905891447139453) q[2];
ry(2.9707386714510315) q[3];
cx q[2],q[3];
ry(0.4520981129636149) q[2];
ry(2.62916919938904) q[3];
cx q[2],q[3];
ry(1.507811485322312) q[3];
ry(-2.4514731242582815) q[4];
cx q[3],q[4];
ry(1.9873352816020624) q[3];
ry(0.14371112576559342) q[4];
cx q[3],q[4];
ry(2.5019321766598583) q[4];
ry(-1.472022532521275) q[5];
cx q[4],q[5];
ry(-3.0428450377215412) q[4];
ry(-0.4187462629695252) q[5];
cx q[4],q[5];
ry(0.3884593530850647) q[5];
ry(3.061562669665806) q[6];
cx q[5],q[6];
ry(-0.24046860588109187) q[5];
ry(-2.0805271014703677) q[6];
cx q[5],q[6];
ry(-1.9517938307407645) q[6];
ry(-0.5391690129898937) q[7];
cx q[6],q[7];
ry(1.0059229485670897) q[6];
ry(1.4308592417964066) q[7];
cx q[6],q[7];
ry(2.2396851887961784) q[0];
ry(-0.848604785273432) q[1];
cx q[0],q[1];
ry(1.6524711422300242) q[0];
ry(-1.085474698364402) q[1];
cx q[0],q[1];
ry(1.4800790901335619) q[1];
ry(-0.7380427002236054) q[2];
cx q[1],q[2];
ry(-2.8027880958677573) q[1];
ry(-0.4964456961657504) q[2];
cx q[1],q[2];
ry(0.565894900317331) q[2];
ry(-0.7840910227232125) q[3];
cx q[2],q[3];
ry(-1.9488791456402699) q[2];
ry(-0.20253008726123445) q[3];
cx q[2],q[3];
ry(0.22833225123913348) q[3];
ry(-0.874004621674934) q[4];
cx q[3],q[4];
ry(0.5846145213243412) q[3];
ry(1.0191640764607566) q[4];
cx q[3],q[4];
ry(-0.1326296973671116) q[4];
ry(1.2482016018638857) q[5];
cx q[4],q[5];
ry(-0.982852249442522) q[4];
ry(2.858375581067575) q[5];
cx q[4],q[5];
ry(-1.4362374607293764) q[5];
ry(1.9449340322201056) q[6];
cx q[5],q[6];
ry(-1.5684426360125363) q[5];
ry(-1.7042455943875068) q[6];
cx q[5],q[6];
ry(0.329774147947771) q[6];
ry(2.2217026138944274) q[7];
cx q[6],q[7];
ry(1.5626028495492803) q[6];
ry(2.541218880062725) q[7];
cx q[6],q[7];
ry(-0.6662248048570012) q[0];
ry(2.390496548339248) q[1];
cx q[0],q[1];
ry(1.0237505609963433) q[0];
ry(-0.7003059707166425) q[1];
cx q[0],q[1];
ry(2.072922778244112) q[1];
ry(0.36537010341086085) q[2];
cx q[1],q[2];
ry(-0.16243456621423408) q[1];
ry(-2.1380194578519607) q[2];
cx q[1],q[2];
ry(2.8737311119031355) q[2];
ry(2.512580854433842) q[3];
cx q[2],q[3];
ry(3.065445324357878) q[2];
ry(-0.3430787223672871) q[3];
cx q[2],q[3];
ry(0.11897805104314152) q[3];
ry(-2.832916986796594) q[4];
cx q[3],q[4];
ry(-0.4869060664332424) q[3];
ry(-2.777105501324545) q[4];
cx q[3],q[4];
ry(-2.022650564791304) q[4];
ry(2.742288529136119) q[5];
cx q[4],q[5];
ry(-1.2134872615439005) q[4];
ry(-2.088327435239644) q[5];
cx q[4],q[5];
ry(0.46357985257391654) q[5];
ry(-2.0919934457477236) q[6];
cx q[5],q[6];
ry(2.4430217667621803) q[5];
ry(-2.815120988328228) q[6];
cx q[5],q[6];
ry(0.6049843761654421) q[6];
ry(0.045218154550020095) q[7];
cx q[6],q[7];
ry(2.322560732484808) q[6];
ry(1.7739680943040934) q[7];
cx q[6],q[7];
ry(2.4500606028990894) q[0];
ry(-2.0680638253863446) q[1];
cx q[0],q[1];
ry(2.5735044758612933) q[0];
ry(0.7066826671822435) q[1];
cx q[0],q[1];
ry(0.4045553300875466) q[1];
ry(0.12910866470409577) q[2];
cx q[1],q[2];
ry(2.0164632904345403) q[1];
ry(-2.3218776184660683) q[2];
cx q[1],q[2];
ry(0.4215341937173589) q[2];
ry(-1.2703673222796006) q[3];
cx q[2],q[3];
ry(-0.4110207180994654) q[2];
ry(2.591779135507712) q[3];
cx q[2],q[3];
ry(2.061895043518763) q[3];
ry(-0.38574876398417274) q[4];
cx q[3],q[4];
ry(0.8441719081648213) q[3];
ry(2.931533761281241) q[4];
cx q[3],q[4];
ry(1.4579976059160478) q[4];
ry(1.064791705725932) q[5];
cx q[4],q[5];
ry(-2.130248478747199) q[4];
ry(-2.6134276628105377) q[5];
cx q[4],q[5];
ry(-1.692206896527904) q[5];
ry(2.8992316179357807) q[6];
cx q[5],q[6];
ry(-0.2605935551747143) q[5];
ry(-0.181823050906952) q[6];
cx q[5],q[6];
ry(-1.6278587172727157) q[6];
ry(-2.0622357566794136) q[7];
cx q[6],q[7];
ry(-2.8193425266893346) q[6];
ry(2.831185897342507) q[7];
cx q[6],q[7];
ry(0.11958279920664248) q[0];
ry(-3.0738266554265112) q[1];
cx q[0],q[1];
ry(2.9006674291074304) q[0];
ry(-1.4560543970876374) q[1];
cx q[0],q[1];
ry(-2.156733913565909) q[1];
ry(-0.16563081356289722) q[2];
cx q[1],q[2];
ry(2.3904681786179363) q[1];
ry(-0.8723327406142714) q[2];
cx q[1],q[2];
ry(-1.681237696225559) q[2];
ry(-1.3455601725259312) q[3];
cx q[2],q[3];
ry(-0.4752608919002638) q[2];
ry(-0.9350688266671641) q[3];
cx q[2],q[3];
ry(-2.089724560364638) q[3];
ry(0.9382919563273048) q[4];
cx q[3],q[4];
ry(1.91263067728218) q[3];
ry(0.8538841361726807) q[4];
cx q[3],q[4];
ry(-2.75777469882922) q[4];
ry(-0.9434785827230385) q[5];
cx q[4],q[5];
ry(2.900404767742293) q[4];
ry(-2.794683560824862) q[5];
cx q[4],q[5];
ry(1.86564440288302) q[5];
ry(-1.272493700758016) q[6];
cx q[5],q[6];
ry(2.907494319795241) q[5];
ry(-2.1068388523654784) q[6];
cx q[5],q[6];
ry(-0.5865835071722197) q[6];
ry(-1.0407797778862262) q[7];
cx q[6],q[7];
ry(-0.27206016783994746) q[6];
ry(-0.6339223015788381) q[7];
cx q[6],q[7];
ry(-3.029568236048292) q[0];
ry(1.7631212833378498) q[1];
cx q[0],q[1];
ry(0.5247922058766123) q[0];
ry(2.981679668429921) q[1];
cx q[0],q[1];
ry(0.5537343123185682) q[1];
ry(-1.0439319480395197) q[2];
cx q[1],q[2];
ry(1.9044707402738892) q[1];
ry(-2.1939692322879725) q[2];
cx q[1],q[2];
ry(-3.027224058465632) q[2];
ry(-0.5062771940289217) q[3];
cx q[2],q[3];
ry(0.9929831806531455) q[2];
ry(0.508037369245093) q[3];
cx q[2],q[3];
ry(-2.1696485150500986) q[3];
ry(2.525335091039544) q[4];
cx q[3],q[4];
ry(2.5047673317897345) q[3];
ry(2.0937299229312356) q[4];
cx q[3],q[4];
ry(0.7160956938602023) q[4];
ry(3.116779120628363) q[5];
cx q[4],q[5];
ry(2.7379998011002216) q[4];
ry(-2.7059976705648925) q[5];
cx q[4],q[5];
ry(2.915630004844191) q[5];
ry(-0.2704246312568257) q[6];
cx q[5],q[6];
ry(-2.650984375546203) q[5];
ry(2.099181674685405) q[6];
cx q[5],q[6];
ry(-0.6103214569404992) q[6];
ry(-2.090799934967193) q[7];
cx q[6],q[7];
ry(-2.73986537477293) q[6];
ry(1.4826527874028512) q[7];
cx q[6],q[7];
ry(-1.563719833358154) q[0];
ry(0.6757599978685498) q[1];
cx q[0],q[1];
ry(-0.8390674100126861) q[0];
ry(1.8621104288292147) q[1];
cx q[0],q[1];
ry(0.08316804776303943) q[1];
ry(0.014502532147217195) q[2];
cx q[1],q[2];
ry(0.9297118504116176) q[1];
ry(-0.05067169635559482) q[2];
cx q[1],q[2];
ry(2.2425775213268104) q[2];
ry(-0.6370647621937782) q[3];
cx q[2],q[3];
ry(1.8639808072528492) q[2];
ry(2.8592229567817404) q[3];
cx q[2],q[3];
ry(0.15038748342087696) q[3];
ry(-2.4704061439342686) q[4];
cx q[3],q[4];
ry(-2.3489822070479156) q[3];
ry(1.0025953442579518) q[4];
cx q[3],q[4];
ry(-2.1553215715673124) q[4];
ry(0.7709381406066909) q[5];
cx q[4],q[5];
ry(2.1051035453475304) q[4];
ry(-1.2797619235618063) q[5];
cx q[4],q[5];
ry(-0.8010354996825183) q[5];
ry(1.8090585899833547) q[6];
cx q[5],q[6];
ry(1.5464422720085411) q[5];
ry(0.9396459746438705) q[6];
cx q[5],q[6];
ry(-0.4337167157112946) q[6];
ry(0.6932466823425887) q[7];
cx q[6],q[7];
ry(2.545738925387481) q[6];
ry(1.7616445473319726) q[7];
cx q[6],q[7];
ry(-1.3248273325643618) q[0];
ry(-2.9039311785909296) q[1];
cx q[0],q[1];
ry(0.1872735956252766) q[0];
ry(-1.4968544171637266) q[1];
cx q[0],q[1];
ry(-2.4394153670669962) q[1];
ry(-2.4879109347200687) q[2];
cx q[1],q[2];
ry(-1.3509828648673092) q[1];
ry(-0.5023361583138249) q[2];
cx q[1],q[2];
ry(-2.6915661769276635) q[2];
ry(-1.2371458493023058) q[3];
cx q[2],q[3];
ry(0.0798914591454829) q[2];
ry(-1.5164326626433349) q[3];
cx q[2],q[3];
ry(1.2754825425161658) q[3];
ry(-1.9764383465925777) q[4];
cx q[3],q[4];
ry(0.5084182628601175) q[3];
ry(1.4787204687976818) q[4];
cx q[3],q[4];
ry(-2.416588681360104) q[4];
ry(-2.687251102449719) q[5];
cx q[4],q[5];
ry(2.2434728514800275) q[4];
ry(-1.9120101947907397) q[5];
cx q[4],q[5];
ry(1.5688447190796042) q[5];
ry(1.8724462802311925) q[6];
cx q[5],q[6];
ry(2.6138832163325874) q[5];
ry(0.5226112330037402) q[6];
cx q[5],q[6];
ry(2.531297625791151) q[6];
ry(1.2062083868299478) q[7];
cx q[6],q[7];
ry(1.6924031865198872) q[6];
ry(-2.50443780271024) q[7];
cx q[6],q[7];
ry(2.1125854222919056) q[0];
ry(1.2658902492770836) q[1];
cx q[0],q[1];
ry(2.2578569014638594) q[0];
ry(-0.8139895371523487) q[1];
cx q[0],q[1];
ry(0.4396897933463979) q[1];
ry(1.275780732736603) q[2];
cx q[1],q[2];
ry(-0.8243177975330251) q[1];
ry(-2.6435853408408327) q[2];
cx q[1],q[2];
ry(0.2527087831500459) q[2];
ry(-3.0978175701089135) q[3];
cx q[2],q[3];
ry(-0.9797920351569939) q[2];
ry(2.945003060903283) q[3];
cx q[2],q[3];
ry(2.96993129788479) q[3];
ry(0.3459257712330768) q[4];
cx q[3],q[4];
ry(-0.0682232506299334) q[3];
ry(-2.3093436267640266) q[4];
cx q[3],q[4];
ry(3.0231533952621827) q[4];
ry(-0.5695790590725454) q[5];
cx q[4],q[5];
ry(2.065010455965292) q[4];
ry(2.12097481702983) q[5];
cx q[4],q[5];
ry(2.98192995149282) q[5];
ry(-1.5266463645688662) q[6];
cx q[5],q[6];
ry(0.7625587620240308) q[5];
ry(-0.15629404077660958) q[6];
cx q[5],q[6];
ry(-3.0810204522754634) q[6];
ry(1.9042142281266086) q[7];
cx q[6],q[7];
ry(-1.5001286041453818) q[6];
ry(-3.1038902092162948) q[7];
cx q[6],q[7];
ry(-0.07224680107994086) q[0];
ry(0.6440521418820682) q[1];
ry(-2.9882014903141574) q[2];
ry(3.0240245715562173) q[3];
ry(1.7245802019721774) q[4];
ry(1.075623736179127) q[5];
ry(1.6400978650602789) q[6];
ry(1.4142810348701076) q[7];