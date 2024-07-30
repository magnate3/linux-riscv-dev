;;; p4_16-mode.el --- Support for the P4_16 programming language

;; Copyright (C) 2016- Barefoot Networks
;; Author: Vladimir Gurevich <vladimir.gurevich@barefootnetworks.com>
;; Maintainer: Vladimir Gurevich <vladimir.gurevich@barefootnetworks.com>
;; Created: 15 April 2017
;; Version: 0.3
;; Keywords: languages p4_16
;; Homepage: http://p4.org

;; This file is not part of GNU Emacs.

;; This file is free software…

;; This mode has preliminary support for P4_16. It covers the core language,
;; but it is not clear yet, how we can highlight the indentifiers, defined
;; for a particular architecture. Core library definitions are included

;; Placeholder for user customization code
(defvar p4_16-mode-hook nil)

;; Define the keymap (for now it is pretty much default)
(defvar p4_16-mode-map
  (let ((map (make-keymap)))
    (define-key map "\C-j" 'newline-and-indent)
    map)
  "Keymap for P4_16 major mode")

;; Syntactic HighLighting

;; Main keywors (declarations and operators)
(setq p4_16-keywords
      '("abstract" "action" "apply"
        "control"
        "default"
        "else" "enum" "extern" "exit"
        "header" "header_union"
        "if"
        "match_kind"
        "package" "parser"
        "return"
        "select" "state" "struct" "switch"
        "table"  "transition" "tuple" "type" "typedef"
        "value_set" "verify"
        ))

(setq p4_16-annotations
      '("@name" "@metadata" "@alias" "@optional"
        "@defaultonly"  "@synchronous"
        ;;; bf-p4c annotations
        "@__intrinsic_metadata"
        "@alpm"
        "@alpm_partitions"
        "@alpm_subtrees_per_partition"
        "@atcam_number_partitions"
        "@atcam_partition_index"
        "@calculated_field_update_location"
        "@command_line"
        "@critical"
        "@deprecated"
        "@disable_atomic_modify"
        "@disable_reserved_i2e_drop_implementation"
        "@dont_unroll"
        "@dontmerge"
        "@dynamic_table_key_masks"
        "@entries_with_ranges"
        "@field_list_field_slice"
        "@flexible"
        "@force_immediate"
        "@idletime_interval"
        "@idletime_per_flow_idletime"
        "@idletime_precision"
        "@idletime_two_way_notification"
        "@ignore_table_dependency"
        "@immediate"
        "@in_hash"
        "@lrt_enable"
        "@max_actions"
        "@max_loop_depth"
        "@min_width"
        "@noWarn"
        "@no_field_initialization"
        "@override_phase0_action_name"
        "@override_phase0_table_name"
        "@pa_alias"
        "@pa_atomic"
        "@pa_auto_init_metadata"
        "@pa_container_size"
        "@pa_container_type"
        "@pa_do_not_bridge"
        "@pa_mutually_exclusive"
        "@pa_no_init"
        "@pa_no_overlay"
        "@pa_not_deparsed"
        "@pa_not_parsed"
        "@pa_solitary"
        "@pack"
        "@padding"
        "@packet_entry"
        "@phase0"
        "@placement_priority"
        "@proxy_hash_algorithm"
        "@proxy_hash_width"
        "@selector_max_group_size"
        "@selector_enable_scramble"
        "@stage"
        "@symmetric"
        "@terminate_parsing"
        "@use_hash_action"
        "@user_annotation"
        "@ways"

        ))

(setq p4_16-attributes
      '("const" "in" "inout" "out"
        ;; Tables
        "key" "actions" "default_action" "entries" "implementation"
        "counters" "meters" "registers" "filters" "size" "idle_timeout"
        "proxy_hash" "requires_versioning" "alpm" "atcam"
        ;; Checksum and others
        "data=" "zeroes_as_ones="
        ;; Parser counter
        "max=" "rotate=" "mask=" "add="
        ;; Hash
        "coeff=" "reversed=" "msb=" "extended=" "init=" "xor="
        "algo="  "poly="
        ;; Stateful (counters, meters, etc.)
        "type="  "index=" "true_egress_accounting=" "adjust_byte_count="
        "red=" "yellow="  "green=" "color="
        "drop_value="     "no_drop_value="
        "initial_value="
        ))

(setq p4_16-variables
      '("next" "last" "this"
        ;;; TNA Externs
        "Alpm"
        "Atcam"
        "Checksum"
        "Counter"  "DirectCounter"
        "Meter"    "DirectMeter"
        "Lpf"      "DirectLpf"
        "Wred"     "DirectWred"
        "Register" "DirectRegister" "RegisterParam" "MathUnit"
        "RegisterAction" "DirectRegisterAction"
        "ActionSelector" "ActionProfile"
        "ParserCounter" "ParserPriority"
        "Hash"     "CRCPolynomial"
        "Random"
        "Mirror" "Resubmit" "Digest"
        ;;; TNA Package Elements
        "Pipeline" "Switch"
        ;;; T2NA Externs
        "IdleTimeout"
        "RegisterAction2" "RegisterAction3" "RegisterAction4"
        "DirectRegisterAction2" "DirectRegisterAction3" "DirectRegisterAction4"
        "LearnAction" "LearnAction2" "LearnAction3" "LearnAction4"
        "MinMaxAction" "MinMaxAction2" "MinMaxAction3" "MinMaxAction4"
        "PktGen"
       ))

(setq p4_16-operations
      '("&&&" ".." "++" "?" ":" "|+|" "|-|"))

(setq p4_16-constants
      '(
        ;;; Don't care
        "_"
        ;;; bool
        "false" "true"
        ;;; error
        "NoError" "PacketTooShort" "NoMatch" "StackOutOfBounds"
        "OverwritingHeader" "HeaderTooShort" "ParserTiimeout"
        ;;; match_kind
        "exact" "ternary" "lpm" "range" "selector" "dleft_hash"
        "atcam_partition_index"
        ;;; We can add constants for supported architectures here
        "CounterRange" "Timeout" "PhvOwner" "MultiWrite"
        "IbufOverflow" "IbufUnderflow"
        ;;;
        "PORT_METADATA_SIZE"
        ))

(setq p4_16-types
      '("bit" "bool" "int" "varbit" "void" "error"
        "packet_in" "packet_out"
        ;;; Special enums that's nice to highlight
        "CounterType_t"   "PACKETS" "BYTES" "PACKETS_AND_BYTES"
        "MeterColor_t"    "GREEN" "YELLOW" "RED"
        "SelectorMode_t"  "FAIR" "RESILIENT"
        "HashAlgorithm_t" "IDENTITY" "RANDOM" "CRC8" "CRC16" "CRC32" "CRC64"
                          "CUSTOM"
        "MathOp_t"        "MUL" "SQR" "SQRT" "DIV" "RSQR" "RSQRT"
        ))

(setq p4_16-primitives
      '(
        ;;; Header methods
        "isValid" "setValid" "setInvalid" "minSizeInBytes" "minSizeInBits"
        "push_front" "pop_front"
        ;;; Table Methods
        "hit" "miss" "action_run"
        ;;; packet_in methods
        "extract" "lookahead" "advance" "length" "port_metadata_unpack"
        ;;; packet_out (also Mirror/Resubmit) methods
        "emit"
        ;;; Known parser states
        "accept" "reject"
        ;;; misc
        "NoAction"
        ;;; TNA Extern Methods
        "sizeInBytes" "sizeInBits"              ;;; Headers (incl. flexible)
        "add" "subtract" "verify" "get" "update";;; Checksum
        "subtract_all_and_deposit"              ;;; Checksum
        "min" "max" "invalidate" "is_validated" ;;; Extern functions
	"funnel_shift_right"                    ;;; Extern functions
        "count"                                 ;;; Counters
        "execute" "execute_log"                 ;;; Meters/Lpfs/Wreds/Registers
        "read"    "write"                       ;;; Registers
        "set" "get" "increment" "decrement"     ;;; ParserCounter/ParserPriority
        "is_zero" "is_negative"                 ;;; ParserCounter
        "pack"                                  ;;; Digest
        ;;; T2NA ExternMethods
        "enqueue" "dequeue" "push" "pop" "overflow" "underflow"
        "address" "predicate" "min8" "max8" "min16" "max16"
        ))

(setq p4_16-cpp
      '("#include"
        "#define" "#undef"
        "#if" "#ifdef" "#ifndef"
        "#elif" "#else"
        "#endif"
        "defined"
        "#line" "#file"))

(setq p4_16-cppwarn
      '("#error" "#warning"))

;; Optimize the strings
(setq p4_16-keywords-regexp    (regexp-opt p4_16-keywords   'words))
(setq p4_16-annotations-regexp (regexp-opt p4_16-annotations     1))
(setq p4_16-attributes-regexp  (regexp-opt p4_16-attributes 'words))
(setq p4_16-variables-regexp   (regexp-opt p4_16-variables  'words))
(setq p4_16-operations-regexp  (regexp-opt p4_16-operations 'words))
(setq p4_16-constants-regexp   (regexp-opt p4_16-constants  'words))
(setq p4_16-types-regexp       (regexp-opt p4_16-types      'words))
(setq p4_16-primitives-regexp  (regexp-opt p4_16-primitives 'words))
(setq p4_16-cpp-regexp         (regexp-opt p4_16-cpp        1))
(setq p4_16-cppwarn-regexp     (regexp-opt p4_16-cppwarn    1))


;; create the list for font-lock.
;; each category of keyword is given a particular face
(defconst p4_16-font-lock-keywords
  (list
   (cons p4_16-cpp-regexp         font-lock-preprocessor-face)
   (cons p4_16-cppwarn-regexp     font-lock-warning-face)
   (cons p4_16-types-regexp       font-lock-type-face)
   (cons p4_16-constants-regexp   font-lock-constant-face)
   (cons p4_16-attributes-regexp  font-lock-builtin-face)
   (cons p4_16-variables-regexp   font-lock-variable-name-face)
   ;;; This is a special case to distinguish the method from the keyword
   (cons "\\.apply"               font-lock-function-name-face)
   (cons p4_16-primitives-regexp  font-lock-function-name-face)
   (cons p4_16-operations-regexp  font-lock-builtin-face)
   (cons p4_16-keywords-regexp    font-lock-keyword-face)
   (cons p4_16-annotations-regexp font-lock-keyword-face)
   (cons "\\(\\w*_[ht]\\)[[< ]"   '(1 font-lock-type-face))
   (cons "[^A-Z_][A-Z] "       font-lock-type-face) ;; Total hack for templates
   (cons "<[A-Z, ]*>"          font-lock-type-face)
   (cons "\\(<[^>]+>\\)"       font-lock-string-face)
   (cons "[^_[:alnum:]]\\(\\([[:digit:]]+w\\)?0b[01_]+\\)"           '(1 font-lock-constant-face))
   (cons "[^_[:alnum:]]\\(\\([[:digit:]]+w\\)?0o[0-7_]+\\)"          '(1 font-lock-constant-face))
   (cons "[^_[:alnum:]]\\(\\([[:digit:]]+w\\)?0d[[:digit:]_]+\\)"    '(1 font-lock-constant-face))
   (cons "[^_[:alnum:]]\\(\\([[:digit:]]+w\\)?0x[[:xdigit:]_]+\\)"   '(1 font-lock-constant-face))
   (cons "[^_[:alnum:]]\\([+-]?\\([[:digit:]]+w\\)?[[:digit:]_]+\\)" '(1 font-lock-constant-face))
   ;;(cons "\\(\\w*\\)"        font-lock-variable-name-face)
   )
  "Default Highlighting Expressions for P4_16")

(defvar p4_16-mode-syntax-table
  (let ((st (make-syntax-table)))
    (modify-syntax-entry  ?_  "w"      st)
    (modify-syntax-entry  ?/  ". 124b" st)
    (modify-syntax-entry  ?*  ". 23"   st)
    (modify-syntax-entry  ?\n  "> b"   st)
    st)
  "Syntax table for p4_16-mode")

;;; Indentation
(defvar p4_16-indent-offset 4
  "Indentation offset for `p4_16-mode'.")

(defun p4_16-indent-line ()
  "Indent current line for any balanced-paren-mode'."
  (interactive)
  (let ((indent-col 0)
        (indentation-increasers "[{(]")
        (indentation-decreasers "[})]")
        )
    (save-excursion
      (beginning-of-line)
      (condition-case nil
          (while t
            (backward-up-list 1)
            (when (looking-at indentation-increasers)
              (setq indent-col (+ indent-col p4_16-indent-offset))))
        (error nil)))
    (save-excursion
      (back-to-indentation)
      (when (and (looking-at indentation-decreasers)
                 (>= indent-col p4_16-indent-offset))
        (setq indent-col (- indent-col p4_16-indent-offset))))
    (indent-line-to indent-col)))

;;; Imenu support
(require 'imenu)
(setq p4_16-imenu-generic-expression
      '(
        ("Controls"      "^[[:space:]]*control[[:space:]]+\\([[:alpha:]_][[:alnum:]_]*\\)"      1)
        ("Externs"       "^[[:space:]]*extern[[:space:]]+\\([[:alpha:]_][[:alnum:]_]+[[:space:]]+\\)?\\([[:alpha:]_][[:alnum:]_]*\\)[[:space:]]*[<({]"  2)
        ("Tables"        "^[[:space:]]*table[[:space:]]+\\([[:alpha:]_][[:alnum:]_]*\\)"        1)
        ("Actions"       "^[[:space:]]*action[[:space:]]+\\([[:alpha:]_][[:alnum:]_]*\\)"       1)
        ("Parsers"       "^[[:space:]]*parser[[:space:]]+\\([[:alpha:]_][[:alnum:]_]*\\)"       1)
        ("Parser States" "^[[:space:]]*state[[:space:]]+\\([[:alpha:]_][[:alnum:]_]*\\)"        1)
        ("Headers"       "^[[:space:]]*header[[:space:]]+\\([[:alpha:]_][[:alnum:]_]*\\)"       1)
        ("Structs"       "^[[:space:]]*struct[[:space:]]+\\([[:alpha:]_][[:alnum:]_]*\\)"       1)
        ("Header Unions" "^[[:space:]]*header_union[[:space:]]+\\([[:alpha:]_][[:alnum:]_]*\\)" 1)
        ))

;;; Cscope Support
(require 'xcscope)

;; Put everything together
(defun p4_16-mode ()
  "Major mode for editing P4_16 programs"
  (interactive)
  (kill-all-local-variables)
  (set-syntax-table p4_16-mode-syntax-table)
  (use-local-map p4_16-mode-map)
  (set (make-local-variable 'font-lock-defaults) '(p4_16-font-lock-keywords))
  (set (make-local-variable 'indent-line-function) 'p4_16-indent-line)
  (setq major-mode 'p4_16-mode)
  (setq mode-name "P4_16")
  (setq imenu-generic-expression p4_16-imenu-generic-expression)
  (imenu-add-to-menubar "P4_16")
  (cscope-minor-mode)
  (run-hooks 'p4_16-mode-hook)
)

;; The most important line
(provide 'p4_16-mode)
