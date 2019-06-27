---
title: os
tags:
  - tags1
  - tags2
categories: cat
date: 2019-06-27 18:59:06
---

## 1. Course

+ [MIT 6.828](<https://pdos.csail.mit.edu/6.828/2018/schedule.html>)
+ [ustc os](<http://staff.ustc.edu.cn/~xlanchen/OperatingSystemConcepts2017Spring/OperatingSystem2017Spring.htm>)
+ [ustc linux操作系统分析](<http://staff.ustc.edu.cn/~xlanchen/ULK2014Fall/ULK2014Fall.html>)
+ [stanford cs240](http://www.scs.stanford.edu/17sp-cs240/syllabus/)
+ [cmu 15410](https://www.cs.cmu.edu/~410/)

## 2. 系统启动

> 现代计算机使用UEFI，可以一次加载超过512B的boot sector。
>
> 以下为JOS启动流程。

+ 计算机通电后先读取BIOS，加载到`960KB~1MB`地址出。
+ BIOS进行硬件自检，同时打印硬件信息在屏幕上。有问题蜂鸣器会响。
+ 之后计算机读取BIOS设置好的优先级最高的外部存储设备。读取第一个扇区（512字节，即主引导记录`MBR`）到`0x7c00`处。如果这个扇区最后两个字节是`0x55`和`0XAA`表明可以启动，否则不能。
+ 主引导记录即`boot.S`，其中的主要流程包括：
  + 关中断，开`A20`线（兼容性问题，寻找范围更大），加载`段表`(`lgdt gdtdesc`)（包含操作系统内核的段信息），寻址方式变为`segment:offset`
  + 设置`CR0`寄存器的`CR0_PE_ON`为1：从实模式（16bit寻址）切换到保护模式（32bit寻址）
    + `实模式`下寻址方式：`物理地址 = 段基址<<4 + 段内偏移`，早期寄存器是16位，地址线是20位，只能访问1MB
      +  例如：`%cs = xff00`，`%ax = 0x0110`，则物理地址为：`0xff00<<4 + 0x0110 = 0xff110`
    + 保护模式下寻址方式：`segment:offset`，segment找到该段的基址base，（检查flags可访问的话）之后直接与offset相加得到线性地址，如果没有分页则线性地址等于物理地址。
      + 程序员看到的是虚拟地址，但指令中出现的是逻辑地址即`segment:offset`
  + 跳转到保护模式代码段，该段主要执行：
    + 设置各寄存器的值，包括cs指令地址，ds数据段地址，ss栈指针等，jos设置其指向`0x7c00`
    + `call bootmain`：即`boot/main.c`中的代码
+ `bootmain`的主要工作：
  + 将硬盘上的`kernel`加载到内存里的`0x10000`处，其中`.text`段在`0x100000`处。并执行`ELFHDR->e_entry`(`0x10000C`处，`boot/entry.S`)。内核映像为ELF格式。
    + entry处的代码主要：这是CR3寄存器，开启分页，设置内核堆栈的起始地址（`0xf0110000`），之后`call  i386_init`

<!-- more -->

## 3. [内存管理](https://edsionte.com/techblog/%E5%86%85%E5%AD%98%E7%AE%A1%E7%90%86)

### 3.1、分页

+ JOS采用二级页表，CR3指向`page dircectory`，虚拟地址高10位查找`page dircectory`中的`Page Table Entry（PTE）`，每个`PTE`存放了一个页地址的高20位，PTE的低12位用作标志位，物理地址的低12位是从原虚拟地址直接拷贝过来的。

+ 现代`X86`是flat模式，即只有一个段，即段基址为0，所以虚拟地址=线性地址。

  + 在平坦模式下，段机制在两个地方会被用上，一个是per-cpu变量（内核开发中使用），另一个是[线程局部存储](http://www.maxwellxxx.com/TLS)（用户态开发中使用），它们分别会用到gs段和fs段。

+ 现代`X86`是四级页表。

+ 32位三级页表

  + PDE和PTE的每一项为32位。PD和PT的大小为4K。

  + 这种模式下仅能访问到4G的物理地址空间，应用程序和内核能用到的线性地址空间为4G。

    ![1560843558017](http://ww1.sinaimg.cn/large/77451733gy1g4fwwl4iyhj20d907jq3f.jpg)

+ 32位PAE模式

  + PDE和PTE的每一项为64位。PD和PT的大小为4K

  + 这种模式下能访问到64G的物理地址空间，应用程序和内核能用到的线性地址空间为4G

    ![1560843571145](http://ww1.sinaimg.cn/large/77451733gy1g4fwwyn6poj20dj089wf3.jpg)

+ X86_64（IA32E/AMD64）下的4级页表

  + PDE和PTE的每一项为64位。PD和PT的大小为4K

  + 理论上，这种模式下能访问到2^64大小的物理地址空间，应用程序和内核能用到的线性地址空间为2^64。实际上，能访问的线性地址空间大小为`2^48`次方。

  + 在linux内核中，这四级页表分别被称为`PGD、PUD、PMD、PT`。

    ![1560843673406](http://ww1.sinaimg.cn/large/77451733gy1g4fwxk0nauj20df0aaaat.jpg)

  + `PTE`项各字段的作用

    + P位，表示PTE指向的物理内存是否存在(Present)，这可以用于标识物理页面是否被换出
    + R/W位，表示这个页是否是可读/可写的。这可以用于fork的cow机制
    + D位，表示这个页是否是脏页。一旦有数据写入到这个页面，CPU会自动将这个位置位。这可以用来在内核做页面回收及换入换出时，判断是否需要将这个页的数据写入到磁盘中。（实际的实现机制不清楚，但理论上是可以的）
    + U/S位，表示这个页面是否能够在用户态下被访问，内核通过此位来识别用户空间或内核空间的页
    + CPL（current privilege level），用两位表示，分别对应0-3共四个级别，放在CS段选择子的低两位，表示当前的运行等级（ring0-ring3）CPL<3为supervisor模式，CPL=3为user模式
    + 在用户态下，访问U/S位为0的页将引起page fault，这用来阻止用户程序对内核空间的非法访问。

    ![1560844318534](http://ww1.sinaimg.cn/large/77451733gy1g4fwxwnv4nj20fe0aywho.jpg)

### 3.2、页框管理

![1560848883315](http://ww1.sinaimg.cn/large/77451733gy1g4fwya12bnj214g0b0n27.jpg)

+ 页框号`pfn`与`struct page`之间的转换，配置了`SPARSEMEM_VMEMMAP`的情况下：

```c
// 此处相当于做了一个优化，加速转换，但是mem_section还是存在的
#if defined(CONFIG_SPARSEMEM_VMEMMAP)   
#define __pfn_to_page(pfn)  (vmemmap + (pfn))
#define __page_to_pfn(page) (unsigned long)((page) - vmemmap)
#elif defined(CONFIG_SPARSEMEM)
#define __page_to_pfn(pg)                   \
({  const struct page *__pg = (pg);             \
    int __sec = page_to_section(__pg);          \
    (unsigned long)(__pg - __section_mem_map_addr(__nr_to_section(__sec))); \
})

#define __pfn_to_page(pfn)              \
({  unsigned long __pfn = (pfn);            \
    struct mem_section *__sec = __pfn_to_section(__pfn);    \
    __section_mem_map_addr(__sec) + __pfn;      \
})
```

```c
#define NR_SECTION_ROOTS  DIV_ROUND_UP(NR_MEM_SECTIONS, SECTIONS_PER_ROOT) //1<<(19/8)=2k
#define SECTIONS_SHIFT  (MAX_PHYSMEM_BITS - SECTION_SIZE_BITS) // 46 - 27 = 19
#define NR_MEM_SECTIONS   (1UL << SECTIONS_SHIFT)  // 1 << 19
#define SECTIONS_PER_ROOT       (PAGE_SIZE / sizeof (struct mem_section))  // 4k/16 = 256
// SECTION_SIZE_BITS 为 27 表示每个 section 为 128M
struct mem_section {
	unsigned long section_mem_map; 
    unsigned long *pageblock_flags;
};

#define PAGES_PER_SECTION       (1UL << PFN_SECTION_SHIFT)  // 1 << 15 = 32k
#define PFN_SECTION_SHIFT (SECTION_SIZE_BITS - PAGE_SHIFT)  // 27 - 12 = 15
// 即每个 section 管理 32k * 4k = 128M 的物理内存
// 每个 struct pages 大小为 64B, 即每个 section 所存的 struct page 数组的大小为 32k*64B = 2M
```

```c
// 全局的，每个 NUMA 节点一个 pglist_data
struct pglist_data *node_data[MAX_NUMNODES] __read_mostly; 
#define MAX_NUMNODES    (1 << NODES_SHIFT) 
#define NODES_SHIFT     CONFIG_NODES_SHIFT   // 自己的服务器上为 10， 阿里云单核为 4 ？

#define MAX_NR_ZONES 5
// MAX_ZONELISTS 为 1, NUMA下 为 2, 本节点的 node_zonelists 不够可使用备用的 node_zonelists 分配
typedef struct pglist_data {
	struct zone node_zones[MAX_NR_ZONES];
	struct zonelist node_zonelists[MAX_ZONELISTS];
	int nr_zones;
    unsigned long node_start_pfn;
	unsigned long node_present_pages; /* total number of physical pages */
	/* total size of physical page range, including holes */
    unsigned long node_spanned_pages;
	int node_id;
}pg_data_t;

struct zone {  
    struct free_area free_area[MAX_ORDER]; // 每个 free_area 即伙伴系统的一个2^i个连续页框的节点
}  

enum zone_type {
    ZONE_DMA,
    ZONE_DMA32,
    ZONE_NORMAL,
    ZONE_HIGHMEM,
    ZONE_MOVABLE,
    ZONE_DEVICE,
    __MAX_NR_ZONES
};

// strcut page 中的 flags 成员中的某些位可以反推出该 page 属于哪一个 node 及哪一个 zone
// 每个zone中都有
```

### 3.3、buddy系统

+ 用于管理页框缓冲池，解决频繁申请不同大小的连续页框导致的外部碎片问题。	
+ 把所有的空闲页框组织为11个块链表：每个链表分别含`2^0~2^10`个连续的空闲页框。
+ 假设要找256个连续空闲页框，先定位到256对应的链表，如果没有再定位到512对应的链表，有的话就分割，并把剩余的256加入256对应的链表，以此类推。

### 3.4、slab分配器

+ 对若干来自于buddy系统的页框块进行有效管理，以满足不同大小“对象”分配的请求。
+ kmalloc_index函数返回kmalloc_caches数组的索引。
+ kmalloc根据对象大小，到相应的缓冲区中进行分配。
+ 当所分配对象大于8KB时，kmalloc维护的13个缓冲区已经无法使用了。
+ 通过调用buddy系统获取页框。
  | kmalloc_caches数组索引 | 对象大小范围（字节） |
  | ---------------------- | -------------------- |
  | 1                      | (64, 96]             |
  | 2                      | (128,   192]         |
  | 3                      | (0,   8]             |
  | 4                      | (8,   16]            |
  | 5                      | (16,   32]           |
  | 6                      | (32,   64]           |
  | 7                      | (96,   128]          |
  | 8                      | (192, 256]           |
  | 9                      | (256,   512]         |
  | 10                     | (512,   1024]        |
  | 11                     | (1024, 2048]         |
  | 12                     | (2048, 4096]         |
  | 13                     | (4096,   8192]       |

### 3.5、kmalloc和vmalloc

+ kmalloc
  + 根据`flags`找到slab中对应的`kmem_cache`（如`__GFP_DMA`，`__GFP_NORMAL`等），并从中分配内存。
  + 分配的内存地址在物理地址和虚拟地址是上都连续。
+ vmalloc
  + 分配的内存地址虚拟地址上连续，在物理地址上不要求连续。
  + 需要建立专门的页表项把物理上不连续的页转换为虚拟地址空间上的页。映射时会出现TLB抖动，因为多次查映射表可能在`TLB`（Translation lookaside buffer，是一种硬件缓冲区）中不断切换。

### 3.6、用户态内存管理：mmap和brk

> 简单来说就是先从`per-thread vma`缓存中找再从红黑树中找满足大小要求的`vm_area_struct`。虚拟地址空间映射好后，物理内存的的分配与映射是通过`page fault`完成的。对于文件映射会先预读数据到内核中。如果指定了`populate`选项会直接先建立到物理内存的分配与映射。匿名映射是先创建一个创建一个临时的`shm file`（包括它的`dentry`，`inode`和`file`结构体）再与合适的`vma`映射。

+ `mmap`的那些范围通过`vm_area_struct`结构体管理。这个结构体以起始范围为key，通过红黑树连接起来，每个相邻的`vm_area_struct`之间同时还使用双向链表连接。
+ 用户态与`mmap`有关的流程：
  + 调用`mmap`，映射一个文件的一部分或者一段内存空间
  + 如果`mmap`的时候要求`populate`，则需要先建立相关的映射（对于文件而言，需要预读数据到内核中，并建立映射关系；对于内存而言，需要映射相关的内存空间）
  + 在用户态的运行过程中，可能会在访问某个地址时出现`page fault`。这时，`page fault`经过层层检查，发现是`mmap`的一段区域引发的，就会去建立映射关系（如上文）。
  + 在建立好映射关系后，重新执行引起`page fault`的指令
  + 调用`msync`或者取消映射时，对于文件而言，需要将数据刷到磁盘上
  + 调用`munmap`取消映射
+ `mmap`内部的实现：
  + 通过`sys_mmap`进入内核（函数实现在[/](https://elixir.bootlin.com/linux/v4.16.18/source)[arch](https://elixir.bootlin.com/linux/v4.16.18/source/arch)/[x86](https://elixir.bootlin.com/linux/v4.16.18/source/arch/x86)/[kernel](https://elixir.bootlin.com/linux/v4.16.18/source/arch/x86/kernel)/[sys_x86_64.c](https://elixir.bootlin.com/linux/v4.16.18/source/arch/x86/kernel/sys_x86_64.c)中） 
  + 调用到了`sys_mmap_pgoff`（函数实现在[/](https://elixir.bootlin.com/linux/v4.16.18/source)[mm](https://elixir.bootlin.com/linux/v4.16.18/source/mm)/[mmap.c](https://elixir.bootlin.com/linux/v4.16.18/source/mm/mmap.c)中）
    + 如果是非匿名映射，获取`fd`对应的`file`。如果`file`是大页文件系统的文件，将`length`对齐到`2M`
    + 否则，如果设置了`MAP_HUGETLB`标志，在大页文件系统中创建一个`file`
    + 调用`vm_mmap_pgoff`
      + 安全检查
      + 调用`do_mmap_pgoff`，这个函数转发数据到`do_mmap`。`do_mmap`的流程见下文。
      + 如果用户指定了要`populate`，则调用`mm_populate`创建具体的映射关系
  + 完成`mmap`，将`file`的引用计数减一
+ `do_mmap`
  + 参数检查
  + 获取未映射的内存区域
    + 如果是文件映射，且`file->f_op->get_unmapped_area`不为`NULL`，则调用该函数搜索未映射的内存区域
    + 如果是共享内存映射，则调用`shmem_get_unmapped_area`搜索未映射的内存区域
    + 如果上述两者都为`NULL`，则调用`current->mm->get_unmapped_area`，该函数一般对应到`arch_get_unmapped_area`（位于[/](https://elixir.bootlin.com/linux/v4.16.18/source)[arch](https://elixir.bootlin.com/linux/v4.16.18/source/arch)/[x86](https://elixir.bootlin.com/linux/v4.16.18/source/arch/x86)/[kernel](https://elixir.bootlin.com/linux/v4.16.18/source/arch/x86/kernel)/[sys_x86_64.c](https://elixir.bootlin.com/linux/v4.16.18/source/arch/x86/kernel/sys_x86_64.c)中）
  + 对其他参数进行检查
  + 调用`mmap_region`进行映射。
+ `mmap_region`
  + 将映射范围内原来的那些已经被映射的空间`unmap`掉
  + 判断能否和前后范围内的`vma`合并，如果能合并，则函数返回
  + 否则，新建一个`vma`并初始化
  + 如果是文件映射，则调用`file->f_op->mmap`来进行具体的映射
  + 否则，如果是匿名共享内存映射，则创建一个临时的`shm file`（包括它的`dentry`，`inode`和`file`结构体），并使用`shmem_file_operations/shmem_inode_operations/shmem_aops`等来初始化`file/inode/address_space`，对于`vma`，使用`shmem_vm_ops`来初始化
  + 接下来，将`vma`加入到红黑树中
+ `brk`也会去把已经映射的区域`unmap`掉，然后再做`mmap`。`brk`其实是一个简陋的`mmap`，它不管文件，`cow`之类的问题，只匿名映射内存。在`malloc`的实现中，有了`mmap`，一般不会去调用`brk`，因为它有可能把原来的映射给释放掉。

### 3.7、内核地址空间共享机制

+ 内核地址空间是共享的，每当`fork`一个进程时，内核会拷贝属于内核地址空间的那部分`pgd`表，使得所有的进程的`pgd`表中关于内核地址空间的部分都指向同样的`pud`表。
+ 但是在运行过程中，内核可能又在`vmalloc`区域映射了一些新的页，而其他进程是不知道的，这些进程在陷内核访问这些页时，会触发`page fault`。在`page fault`的处理例程中，会去`init_mm`中查找对应区域的这些页表项，将这些页表项拷贝到触发`page fault`的进程的页表项中。（事实上，只有`pgd`表项为空，才会触发`vmalloc`的`page fault`）
+ 对于内核`module`来说，如果在运行时动态加载了一个`module`，那么它的代码会被加载到一个专门的区域中，但这个区域并不在`vmalloc`区域内，那么内核中的其他进程怎么知道这一新映射的`module`代码的位置呢？事实上，这一区域能够被一个`pgd`表项`cover`住，那么，在`fork`的时候，所有进程的这一个`pgd`表项，已经对应到了同一个对应的`pud`页。内核在映射`module`的时候，是修改的`pud/pmd`等页表项，其他进程自然能够看到这一映射关系，而不会引发`page fault`

### 3.8、共享内存

+ 这一套是在`glibc`里面实现的
  + `shm_open`创建一个`file`，获取对应的`fd`
  + `mmap`映射`fd`
  + `munmap`取消映射
  + `shm_unlink`减少引用计数
+ 这一套机制是通过`open`在`/dev/shm`下面创建文件，用`mmap`来映射的。这套机制会经过`fdtable`，所以更为安全。

### 3.9、 ptmalloc

+ `ptmalloc`通过`mmap`或brk（仅`main_arena`使用）每次批发`64MB`大小的内存（称为`heap`）来管理。
+ `ptmalloc`中有多个`arena`，即`struct malloc_state`。只有一个main_arena，其它的`arena`是动态分配的，但数量有上限，这些`arena`以链表的形式组织。分配内存找`arena`时可能会逐个对空闲的`arena`进行`try_lock`。
+ 每个`arena`管理多个`heap`。每个`heap`通过`prev`指针连接起来，`arena->top`指向最近分配的`heap`中的，未分配给用户的，包含了`heap`尾部的那个`chunk`。通过`arena->top`，能够找到最近的那个`heap`。通过最近的那个`heap`的`prev`指针，能够依次找到以前所有的`heap`。
+ `heap`释放时会出现某一个`heap未`全部空闲则该heap前面的空闲的`heap`无法得到释放的问题。
+ 用户请求的空间都用`chunk`来表示。
+ 所有的空闲的chunk被组织在`fastbin`，`unsorted bin`，`small bin`，`large bin`中。
+ `fastbin`可看作`small bin`的缓存，缓存`16B~64B`（以`8B`递进，共`10`个）的`chunk`。
+ `unsorted bin`，`small bin`，`large bin`都是在bins数组中，只是不同`index`范围区分了他们。
+ `chunk`被分配出去时，其在空闲链表中的前后指针以及前后`chunk size`的字段会被复用给用户作为空闲区域。
+ 每个线程有`tcache`，并且每个线程有一个`thread_arena`指向当前线程正在使用的`arena`。
+ `unsorted bin`可以看作`small bin`和`large bin`的缓冲，链表头是`bins[1]`。`fastbin`中的空闲`chunk`合并时会先放到`unsorted bin`中。分配时检查`unsorted bin`没有合适的chunk就会将`unsorted bin`中的chunk放到`small bin`或`large bin`中。
+ `small bin`中的chunk大小范围为`32B~1088B`（以`16B`递增，共`62`个`bin`链表，`bin[2]~bin[63]`）。
+ `large bin`中的chunk大小范围为`1024B`以上。
+ 另外，每个线程有一个`tcache`，共64项，从最小的32字节开始，以`16`字节为单递增。
+ 链表操作通过`CAS`，`arena`争用需要加锁。

### 3.10、[jemalloc](https://youjiali1995.github.io/allocator/jemalloc/)

+ `jemalloc` 中大量使用了宏生成代码，比较晦涩。
  + 通过避免 `false cache line sharing`，使用内存着色等，提高 `cache line` 效率
  + 使用多个 `arena` 管理、更细粒度的锁、 `tsd`、`tcache`等，最小化锁竞争
  + 使用 `slab` 分配不同大小的对象，精心选择 `size classes`，减少内存碎片
  + 使用多层缓存，内存的释放和分配会经历很多阶段，提升速度
+ 每个线程有一个`thread specific data`即 `struct tsd_s tsd`，其中有两个指针，`iarena`和`arena`分别指向用于元数据分配和用于普通数据分配的`arena`。所以，`tsd`需要找两个`arena`进行绑定（这两个`arena`可能是同一个）。
+ `jemalloc`会创建多个`arena`，每个线程由一个 `arena` 负责。`arena`有引用计数字段`nthreads[2]`，`nthreads[0]`计录普通数据分配`arena`指针绑定这个`arena`的次数，`nthreads[1]`记录元数据分配`iarena`指针绑定这个`arena`的次数。一个`tsd`绑定`arena`后，就不会改变`arena`。
+ 有一个全局变量`narenas_auto`，它在初始化时被计算好，表示能够创建的`arena`的最大数量。
+ 有多个`arena`，以全局数组的形式组织。每个线程有一个`tcache`，其中有指向某个`arena`的指针。
+ 当需要绑定一个`arena`时，遍历所有已创建的`arena`，并保存所有`arena`中`nthreads`值最小的那个（根据是绑定元数据还是普通数据，判断使用`nthreads[0]`还是`nthreads[1]`）。
  + 如果在遍历途中发现数组中有`NULL`（也就是说数组有`slot`，还可以创建新的`arena`），那么就创建一个新的`arena`，将`arena`放到那个`slot`中，并绑定在那个`arena`上
  + 如果遍历途中发现所有的`slot`都被用上了，那么就选择`nthreads`值最小的那个，绑定那个`arena`
+ `jemalloc`通过`mmap`以`chunk`（默认2M）为单位向操作系统申请内存。每个`chunk`的头部会有一个`extent_node_t`记录其元数据信息，如所属的`arena`和起始地址。这些`chunk`会以基数树的形式组织起来，保存 `chunk` 地址到 `extent_node_t` 的映射
+ `jemalloc`内部动态分配的内存通过`base`组织。`base` 使用 `extent_node_t` 组成的红黑树 `base_avail_szad` 管理 `chunk`。每次需要分配时，会从红黑树中查找内存大小相同或略大的、地址最低的 `node`， 然后从 `node` 负责的 `chunk` 中分配内存。
+ `chunk` 使用 `Buddy allocation` 划分为不同大小的 `run`。`run` 使用 `Slab allocation` 划分为固定大小的 `region`，大部分内存分配直接查找对应的 `run`，从中分配空闲的 `region`，释放就是标记 `region` 为空闲。
+ `jemalloc` 将对象按大小分为3类，不同大小类别的分配算法不同:
  - `small`（`8B-14K`）: 从对应 `bin` 管理的 `run` 中返回一个 `region`
  - `large（`16K-1792K`）`: 大小比 `chunk` 小，比 `page` 大，会单独返回一个 `run`
  - `huge（`2M-64M`）`: 大小为 `chunk` 倍数，会分配 `chunk`
+ `mutex` 尽量使用 `spinlock`，减少线程间的上下文切换

## 4. 进程管理

+ [`0`号进程](https://blog.csdn.net/gatieme/article/details/51484562)是`idle`进程，`1`号是`init`即 `systemd`进程，`2`号是`kthreadd`进程
+ 当没有进程需要调度器调度时，就执行`idle`进程，其`task_struct`通过宏初始化，存放在内核`data`段
+ 内核`init_task`即`idle`进程的栈大小：`x64`为`16k`，`x86_32`为`8k`
+ `task_struct`内部存一个`void *stack`指向栈的基址

## 5.  地址空间

> **0000000000000000 - 00007fffffffffff (=47 bits) user space, different per mm**
> hole caused by [47:63] sign extension
> ffff800000000000 - ffff87ffffffffff (=43 bits) guard hole, reserved for hypervisor
> **ffff880000000000 - ffffc7ffffffffff (=64 TB) direct mapping of all phys. memory**
> ffffc80000000000 - ffffc8ffffffffff (=40 bits) hole
> **ffffc90000000000 - ffffe8ffffffffff (=45 bits) vmalloc/ioremap space**
> ffffe90000000000 - ffffe9ffffffffff (=40 bits) hole
> **ffffea0000000000 - ffffeaffffffffff (=40 bits) virtual memory map (1TB)**   //vmemmap，存放所有struct page*

+ 用户地址空间

  > 从低地址到高地址：
  >
  > - text  代码段 —— 代码段，一般是只读的区域;
  > - static_data 段 =  
  > - stack 栈区 —— 局部变量，函数的参数，返回值等，由编译器自动分配释放;
  > - heap 堆区 —— 动态内存分配，由程序员分配释放;
  
+ 用户空间和内核空间
    + CPU其实并不知道什么用户空间和内核空间，它是通过PTE的U/S位与CPL来判断这个页是否可以被访问。所以，内核空间的那些页面对应的PTE与用户空间对应的PTE中，U/S位实际上是不同的，内核通过这一方式来划分用户空间和内核空间。
    + 用户态显然不可以访问内核空间，但是内核态下也不一定就能访问用户空间。这与CPU的配置有关，规则很复杂，可以参考Intel手册卷三4.6节，这里提一个关键字SMAP，有兴趣可以自行搜索，在较新的内核版本中，这一机制已经被使用。
    + 关于[vdso和vvar](http://readm.tech/2016/09/23/syscall/)：主要将部分安全的内核代码映射到用户空间，这使得程序可以不进入内核态直接调用系统调用。

    ![1560844894694](http://ww1.sinaimg.cn/large/77451733gy1g4fwys48n0j210s0k646v.jpg)

    + 线性地址空间划分如上图所示。内核空间的划分是确定的，写在内核代码中的，而用户态的空间在可执行文件被装载时才知道，由装载器和链接器来决定（可能需要参考elf相关的文档，才知道具体的装载位置）。不过，用户态空间整体的布局如上图所示，内核的current->mm结构体中记录着相关段的位置。
+ [`task_struct`](<https://elixir.bootlin.com/linux/v4.18/source/include/linux/sched.h#L593>)

+ [`thread_info`](https://blog.csdn.net/gatieme/article/details/51577479)

+ [`task_struct->active_mm`](<http://www.wowotech.net/process_management/context-switch-arch.html>)

  ```c++
  typedef unsigned long	pgdval_t;
  typedef struct { pgdval_t pgd; } pgd_t;
  
  struct mm_struct {
      // 管理mmap出来的内存，brk出来的也是在这brk就是一个简化版的 mmap, 只映射匿名页
      struct vm_area_struct *mmap;   
      struct rb_root mm_rb;       	// 通过红黑树组织
      u32 vmacache_seqnum;                   /* per-thread vmacache */
      pgd_t * pgd;                // 页表指针，与 arch 相关, x64 下 pgd_t 为 unsigned long
  };
  
  struct task_struct {
  	struct thread_info    thread_info; // 位于栈顶部，可用来定位 task_struct
  	void *stack;   // 指向栈
      struct mm_struct *mm, *active_mm;   // 内核线程没有 mm, 但有 active_mm
     	// active_mm 用于 context_switch 时的加速
      
      // 每个vma表示一个虚拟内存区域
      // VMACACHE_SIZE 为 4，per-thread vma缓存
      // 4.4 为 struct vm_area_struct *vmacache[VMACACHE_SIZE]; 
      // 通过 #define VMACACHE_HASH(addr) ((addr >> PAGE_SHIFT) & VMACACHE_MASK)
     	// 决定vma的数组存储位置，PAGE_SHIFT 为 12，VMACACHE_MASK 为 VMACACHE_SIZE-1 即 3
      struct vmacache     vmacache; 
  
  };
  
	struct vm_area_struct {
  // [vm_start, vm_end)
  	unsigned long vm_start;		/* Our start address within vm_mm. */
  	unsigned long vm_end;   // The first byte after our end address within vm_mm.
  	unsigned long vm_flags;		/* 设置读写等权限和是否为共享内存等等，见 mm.h */
  	/* linked list of VM areas per task, sorted by address */
  	struct vm_area_struct *vm_next, *vm_prev; // 2.6 只有 next
      struct rb_node vm_rb;
      struct mm_struct *vm_mm;	/* The address space we belong to. */
      /* Function pointers to deal with this struct. */
  	const struct vm_operations_struct *vm_ops;
      struct file * vm_file;		/* File we map to (can be NULL). */
      /* Information about our backing store: */
  	unsigned long vm_pgoff;		/* Offset (within vm_file) in PAGE_SIZE units */
  };
  
  struct vmacache {
  	u32 seqnum; // 4.4放在 task_struct 中, 当其与 mm_struct 中的 seqnum 不一致时缓存失效
  	struct vm_area_struct *vmas[VMACACHE_SIZE];
  };
  ```
  

## 6. 文件系统

## 7. 系统调用

## 8. 信号

## 9. IO管理

## 10. 中断和异常

### 1、[中断、异常和陷入](https://www.cnblogs.com/johnnyflute/p/3765008.html)

+ `中断`：是为了设备与CPU之间的通信。
  + 典型的有如服务请求，任务完成提醒等。比如我们熟知的时钟中断，硬盘读写服务请求中断。
  + 断的发生与系统处在用户态还是在内核态无关，只决定于`EFLAGS`寄存器的一个标志位。我们熟悉的sti, cli两条指令就是用来设置这个标志位，然后决定是否允许中断。
  + 中断是异步的，因为从逻辑上来说，中断的产生与当前正在执行的进程无关。
+ 异常：异常是由当前正在执行的进程产生。异常包括很多方面，有出错（fault），有陷入（trap），也有可编程异常（programmable exception）。
  + 出错（fault）和陷入（trap）最重要的一点区别是他们发生时所保存的EIP值的不同。出错（fault）保存的EIP指向触发异常的那条指令；而陷入（trap）保存的EIP指向触发异常的那条指令的下一条指令。
  + 因此，当从异常返回时，出错（fault）会重新执行那条指令；而陷入（trap）就不会重新执行。
  + `缺页异常`（page fault），由于是fault，所以当缺页异常处理完成之后，还会去尝试重新执行那条触发异常的指令（那时多半情况是不再缺页）。
  + `陷入`的最主要的应用是在调试中，被调试的进程遇到你设置的断点，会停下来等待你的处理，等到你让其重新执行了，它当然不会再去执行已经执行过的断点指令。
  + 关于异常，还有另外一种说法叫[软件中断（software interrupt](https://blog.51cto.com/noican/1361087)），其实是一个意思。

### 2、中断向量表

+ `IDTR`寄存器指向中断向量表。

## 11. 进程间通信

## 12. 网络

## 13.内核同步机制

## 问题

+ 内核`RCU`机制
+ `seq_lock`：写友好
+ `rcu`：读友好，更新时先拷贝，设置拷贝项的先修改再设置自身指向的前后指针，再设置前面的指向它，再设置自己指向后面的。删除时，与更新相同，只是最后释放的时候调用`synchronize_rcu()`或`call_rcu()`在一段间隔后回收，间隔`(grace period)`的确定是等待当前时间点所有未完成的`read`操作执行完
+ `madvise`
+ 平常调用`mmap`实际调的是`glibc`的`mmap`，其对内核的系统调用`mmap`进行了封装，最终调用的是`mmap2 -> do_mmap`
+ `sys_mmap`源码应当搜索`SYSCALL_DEFINE6(mmap`，里面调用`SYSCALL_DEFINE6(mmap_pgoff`，再进入`ksys_mmap_pgoff -> vm_mmap_pgoff -> do_mmap_pgoff -> do_mmap`
+ `mmap`的最初起始位置有随机页偏移
+ 内核配置一般是没有开启`CONFIG_PREEMPT`，只开启了`CONFIG_PREEMPT_VOLUNTARY`
+ 内核的各种[中断和异常](http://guojing.me/linux-kernel-architecture/posts/interrupt/)
+ [缺页异常的处理](https://edsionte.com/techblog/archives/4174)
+ [内存碎片](http://www.wowotech.net/memory_management/memory-fragment.html)
+ `CPU`执行一条指令的过程：取指、译码、执行、访存、写回
+ [CPU指令乱序](https://zhuanlan.zhihu.com/p/45808885)：
  + cpu为了提高流水线的运行效率，会做出比如：1)对无依赖的前后指令做适当的乱序和调度；2)对控制依赖的指令做分支预测；3)对读取内存等的耗时操作，做提前预读；等等。以上总总，都会导致指令乱序的可能。
  + 但是对于x86的cpu来说，在单核视角上，其实它做出了Sequential consistency的一致性保障，指令在cpu核内部确实是乱序执行和调度的，但是它们对外表现却是顺序提交的，cpu只需要把内部真实的物理寄存器按照指令的执行顺序，顺序映射到ISA寄存器上，也就是cpu只要将结果顺序地提交到ISA寄存器，就可以保证Sequential consistency。
+ per-cpu变量和线程局部存储。




