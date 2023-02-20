OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.33139840825290656) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.4308504478611839) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.08552552662408373) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.33922811955143606) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.3592648557672144) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.13726832191410385) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.243774024171713) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.4342560663005122) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.042283810704129905) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.15460356375844056) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(0.2518781202313438) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-0.03280101958526214) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(0.22149910102947756) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(-0.2383899848993285) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(0.009472721500662586) q[3];
cx q[1],q[3];
rx(-0.18185584785194456) q[0];
rz(-0.13721030218822014) q[0];
rx(0.45464092462750816) q[1];
rz(-0.12720329789934054) q[1];
rx(0.13445114198959335) q[2];
rz(0.0760367808013085) q[2];
rx(-0.529748657945807) q[3];
rz(0.06188712090682574) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.3120629531028073) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.5067969453849248) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.12636097185031459) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.2382639803307021) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.23798059824584128) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.19845699730356206) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.08617104796393187) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.29965550586848616) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.08959866977667416) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.20317214879431753) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(0.4777498986228095) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(0.028084851059466995) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(0.08993450265123583) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(-0.12189742867243508) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(-0.056482703386707196) q[3];
cx q[1],q[3];
rx(-0.20585006156260693) q[0];
rz(-0.0946655026344783) q[0];
rx(0.165079357897714) q[1];
rz(-0.12867789176968855) q[1];
rx(0.020899687458092717) q[2];
rz(-0.08798547659501663) q[2];
rx(-0.24812905572547161) q[3];
rz(-0.05412835211415835) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.3516164869700885) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.5503166093587358) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.2870676943968492) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.1417100938776708) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.11054679607710055) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.18123894159625795) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.22020220570175805) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.19012721943994199) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.034456137908039364) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.37509877792361696) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(0.24380224745705853) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(0.07001881585769518) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.014877250281513732) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(0.08735770430788292) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(-0.014609278569715547) q[3];
cx q[1],q[3];
rx(-0.1255029814389557) q[0];
rz(-0.09864721199336578) q[0];
rx(0.14055106412489246) q[1];
rz(0.04272680184333273) q[1];
rx(-0.13659124342176907) q[2];
rz(0.1048179057444966) q[2];
rx(-0.0212220631014666) q[3];
rz(-0.08043511235943421) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.349328933644647) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.305674265678537) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.19617213025325075) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.05807258527974097) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.32542646767492395) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.06844233059806021) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.3462645128472013) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.0787906143382367) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.02380876983011457) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.3577200327393055) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(0.01561534228434143) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(0.02321452941999594) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.09095768283481298) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(-0.04200969291966298) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(-0.01292792081968374) q[3];
cx q[1],q[3];
rx(-0.09574419856089085) q[0];
rz(-0.0870587610696392) q[0];
rx(0.290881001013826) q[1];
rz(0.21394782525326606) q[1];
rx(-0.05141045843977381) q[2];
rz(0.19168641537115028) q[2];
rx(0.16656430935995215) q[3];
rz(-0.11776363799616016) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.33880977991449496) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.18732523315803862) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.15444670895869692) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.011374197496701497) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.5238597844627227) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.11086216515213863) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.367676234734646) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.1291689785410763) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.06318260709806757) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.41589611614949945) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.09463822855551851) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-0.09830060352137342) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.11081387923171154) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(-0.1496463510090755) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(-0.22141133680626487) q[3];
cx q[1],q[3];
rx(-0.039168880263271574) q[0];
rz(-0.14158841685054416) q[0];
rx(0.19764003214779255) q[1];
rz(0.20621884236758423) q[1];
rx(0.018701208621925796) q[2];
rz(0.16353186265009328) q[2];
rx(0.29828327358754936) q[3];
rz(-0.37280245969148806) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.19163888762168377) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.04604405475007503) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.18041990434001628) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.025080179052718733) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.7232505151954794) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.09315677950804943) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.3939735549048103) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.30244177950994294) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.019465959633482327) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.2624770598343471) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.12454895581999227) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-0.21134150255558784) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.10700650341687205) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(-0.03732043614319904) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(-0.37023792169825687) q[3];
cx q[1],q[3];
rx(-0.00023757713838510843) q[0];
rz(-0.17459517438862146) q[0];
rx(-0.03517737379735241) q[1];
rz(0.06037103436253318) q[1];
rx(0.045871108635797006) q[2];
rz(0.000288445013636986) q[2];
rx(0.3046761833791176) q[3];
rz(-0.3866223499820644) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.013213868774111652) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.2559882725802423) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.2611259751830108) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.20892758699004163) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.8527634813083268) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.18348721813234534) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.41095817104379606) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.2443275168665632) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.15333384466168215) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.011192907242205808) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.10339326795785975) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-0.2515354163202789) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.06482154378750599) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(0.09774687407583246) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(-0.2685738921362955) q[3];
cx q[1],q[3];
rx(0.0938742682574059) q[0];
rz(-0.2112483587521754) q[0];
rx(0.004831979976040487) q[1];
rz(-0.30213180729041306) q[1];
rx(0.006575005151039153) q[2];
rz(0.08399307417963198) q[2];
rx(0.2761218714335749) q[3];
rz(-0.11954779948156198) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.12272825614259872) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.332368358249767) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.25874676177943934) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.38500564263455417) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.9045720847363736) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.29589669678950115) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.48829044092840945) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.19062435811897394) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.1948889713575611) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(0.13383992207023032) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.09380575343643138) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-0.168452910108593) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.07381130884363188) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(-0.18782597165504542) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(-0.31945985098187624) q[3];
cx q[1],q[3];
rx(0.07344049604155091) q[0];
rz(-0.13061666681976328) q[0];
rx(-0.06208130012932672) q[1];
rz(-0.4633821455910887) q[1];
rx(0.15536989589807845) q[2];
rz(-0.019389593060594564) q[2];
rx(0.27083969782718786) q[3];
rz(0.09085968514399677) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.13176333448909627) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.3846568555485532) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.13833542403885032) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.1549509721779189) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.770996791907155) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.01036151694227895) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.24882955128576) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.008859798827514224) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.23987867683747607) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(0.2327745893812588) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.15307002182212204) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-0.24382786212070026) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(0.07332254417837725) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(-0.26260990605549034) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(-0.18316514872376766) q[3];
cx q[1],q[3];
rx(0.023922247769510758) q[0];
rz(-0.18230127533435073) q[0];
rx(-0.06535069095646137) q[1];
rz(-0.426108890225131) q[1];
rx(-0.19385168233831696) q[2];
rz(-0.15457359995382863) q[2];
rx(0.14305539076743337) q[3];
rz(0.2442968092157401) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.24133871739394136) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.44463344283965023) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.13719768718838068) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.3859458022490959) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.05660268894532738) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.32344179134535705) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.14085507628121405) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.19081338685158725) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.10948656508811104) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(0.16639049985384907) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.2955021832674707) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-0.09732253998138322) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(0.035529230956240936) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(-0.2678718640087665) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(0.008223390298661105) q[3];
cx q[1],q[3];
rx(-0.16902538760687605) q[0];
rz(-0.13172646722084824) q[0];
rx(-0.07115720411779451) q[1];
rz(-0.3042798739330325) q[1];
rx(-0.2915015257850381) q[2];
rz(-0.19647536883286645) q[2];
rx(0.10943753569188448) q[3];
rz(0.47402025809128123) q[3];