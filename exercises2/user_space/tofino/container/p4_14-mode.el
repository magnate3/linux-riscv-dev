;;; p4_14-mode.el --- Support for the P4_14 programming language

;; Copyright (C) 2016- Barefoot Networks
;; Author: Vladimir Gurevich <vladimir.gurevich@barefootnetworks.com>
;; Maintainer: Vladimir Gurevich <vladimir.gurevich@barefootnetworks.com>
;; Created: 21 Jan 2016
;; Version: 1.0
;; Keywords: languages p4_14
;; Homepage: http://p4_14.org

;; This file is not part of GNU Emacs.

;; This file is free software…

;; Placeholder for user customization code
(defvar p4_14-mode-hook nil)

;; Define the keymap (for now it is pretty much default)
(defvar p4_14-mode-map
  (let ((map (make-keymap)))
    (define-key map "\C-j" 'newline-and-indent)
    map)
  "Keymap for P4_14 major mode")

;; Syntactic HighLighting

;; Main keywors (declarations and operators)
(setq p4_14-keywords 
      '("action" "action_profile" "action_selector" "apply" "attribute"
        "blackbox" "blackbox_type"
        "calculated_field" "counter" "control" "default" 
        "else" "extern" "extern_type" "extract" 
        "field_list" "field_list_calculation" "header" "header_type" 
        "if" "hit" "metadata" "meter" "method" "miss"
        "parser" "parser_drop" "parser_exception" "parser_value_set"
        "payload"
        "register" 
        "return" "select" "table"
        ))

(setq p4_14-pragma
      '("@pragma")
      )

(setq p4_14-attributes
      '("actions" "algorithm" "attributes"
        "direct"  "dynamic_action_selection" 
        "fields"
        "default_action" "support_timeout"
        "size" "max_size" "min_size"
        "in" "inout" "out" "input" "instance_count" 
        "layout" "output_width" "reads" 
        "update" "verify"
        "width" "min_width"
        "optional"
        "pre_color" "result"
        "static"
        "type"
        "reg"
        "writes"
        "expression_local_variables"
        ;;; action selectors
        "selection_key" "selection_mode"
        ;;; stateful_alu_blackbox-specific attributes
        "update_lo_1_value" "update_lo_2_value"
        "update_hi_1_value" "update_hi_2_value"
        "update_lo_1_predicate" "update_lo_2_predicate"
        "update_hi_1_predicate" "update_hi_2_predicate"
        "output_value" "output_predicate" "output_dst"
        "initial_register_lo_value" "initial_register_hi_value"
        "selector_binding"
        "math_unit_input" "math_unit_output_scale"
        "math_unit_exponent_shift" "math_unit_exponent_invert"
        "math_unit_lookup_table"
        "reduction_or_group"
        "stateful_logging_mode"        
        ;;; lpf-specific attributes
        "filter_input"
        ;;; meter-specific attributes
        "green_value" "yellow_value" "red_value"
        ;;; wred-specific attributes
        "wred_input"        
        ))

(setq p4_14-variables
      '(
        ;;; stateful_alu_blackbox-specific variables
        "alu_lo" "alu_hi"
        "register_lo" "register_hi"
        "math_unit"
        "condition_lo" "condition_hi" "predicate" "combined_predicate"
        "output_value" "output_predicate" "output_dst"
        ;;; lpf-specific variables
        "filter_input"
        ;;; wred-specific attributes
        "wred_input" "drop_value" "no_drop_value"))

(setq p4_14-operations
      '("and" "or" "not" "mask"))

(setq p4_14-constants
      '("false" "true"
        "exact" "ternary" "valid" "range" "lpm"
        "bytes" "packets" "packets_and_bytes"
        "resilient" "fair"
        ;;; Parser pseudo-constants
        "last" "latest" "next"
        ;;; Hash Algorithms
        "identity" "identity_msb" "identity_lsb"
        "csum16"
        "random"
        "crc16" "crc16_msb" "crc16_lsb" "crc16_extend"
        "crc32" "crc32_msb" "crc32_lsb" "crc32_extend"
        "crc_8" "crc_8_darc" "crc_8_i_code" "crc_8_itu"
        "crc_8_maxim" "crc_8_rohc" "crc_8_wcdma"
        "crc_16" "crc_16_buypass" "crc_16_dds_110" "crc_16_dect"
        "crc_16_dnp" "crc_16_en_13757" "crc_16_genibus" "crc_16_maxim"
        "crc_16_mcrf4xx" "crc_16_riello" "crc_16_t10_dif" "crc_16_teledisk"
        "crc_16_usb"
        "x_25" "xmodem" "modbus" "kermit"
        "crc_ccitt_" "crc_aug_ccitt" 
        "crc_32" "crc_32_bzip2" "crc_32c" "crc_32d" "crc_32_mpeg" "crc_32q"
        "jamcrc" "posix" "xfer"
        "crc_64" "crc_64_we" "crc_64_jones"
        ))

(setq p4_14-types
      '("bit" "int" "saturated" "signed" "string" "expression"
        ; Blackbox types
        "stateful_alu" "lpf" "wred"))

(setq p4_14-primitives
      '("add_header" "remove_header" "copy_header" "push" "pop"
        "no_op" "drop"
        "add" "add_to_field" "subtract" "subtract_from_field"
        "bit_and" "bit_andca" "bit_andcb"
        "bit_or"  "bit_orca"  "bit_orcb"
        "bit_xor" "bit_xnor"  "bit_nand" "bit_nor"
        "bit_not"
        "shift_right" "shift_left"
        "max" "min"
        "bypass_egress"
        "exit"
        "funnel_shift_right"
        "invalidate"
        "mark_for_drop"
        "max" "min"
        "modify_field"
        "modify_field_conditionally"
        "modify_field_with_hash_based_offset"
        "modify_field_with_rng_uniform"
        "modify_field_with_shift"
        "swap"
        "resubmit"
        "clone_ingress_pkt_to_egress" "clone_i2e"
        "clone_egress_pkt_to_egress"  "clone_e2e"
        "recirculate"
        "generate_digest"
        "sample_e2e"
        "set_metadata"
        "current"
        "count" "count_from_hash"
        "execute_meter" "execute_meter_from_hash"
        "execute_meter_with_or" "execute_meter_from_hash_with_or"
        ;; Stateful ALUs
        "execute_stateful_alu" "execute_stateful_alu_from_hash"
        "execute_stateful_log"
        "read_bit" "read_bitc" "set_bit" "set_bitc" "clr_bit" "clr_bitc"
        ;; LPF
        "execute"
        ;; Meters
        "execute_with_or" "execute_from_hash" "execute_from_hash_with_or"
        "execute_with_pre_color" "execute_with_pre_color_from_hash"
        "execute_with_pre_color_from_hash_with_or"
        ))

(setq p4_14-cpp
      '("#include" 
        "#define" "#undef"
        "#if" "#ifdef" "#ifndef"
        "#elif" "#else"
        "#endif"
        "defined"
        "#line" "#file"))

(setq p4_14-cppwarn
      '("#error" "#warning"))

;; Optimize the strings
(setq p4_14-keywords-regexp   (regexp-opt p4_14-keywords   'words))
(setq p4_14-pragma-regexp     (regexp-opt p4_14-pragma     1))
(setq p4_14-attributes-regexp (regexp-opt p4_14-attributes 'words))
(setq p4_14-variables-regexp  (regexp-opt p4_14-variables  'words))
(setq p4_14-operations-regexp (regexp-opt p4_14-operations 'words))
(setq p4_14-constants-regexp  (regexp-opt p4_14-constants  'words))
(setq p4_14-types-regexp      (regexp-opt p4_14-types      'words))
(setq p4_14-primitives-regexp (regexp-opt p4_14-primitives 'words))
(setq p4_14-cpp-regexp        (regexp-opt p4_14-cpp        1))
(setq p4_14-cppwarn-regexp    (regexp-opt p4_14-cppwarn    1))


;; create the list for font-lock.
;; each category of keyword is given a particular face
(defconst p4_14-font-lock-keywords
  (list
   (cons p4_14-cpp-regexp         font-lock-preprocessor-face)
   (cons p4_14-cppwarn-regexp     font-lock-warning-face)
   (cons p4_14-types-regexp       font-lock-type-face)
   (cons p4_14-constants-regexp   font-lock-constant-face)
   (cons p4_14-attributes-regexp  font-lock-builtin-face)
   (cons p4_14-variables-regexp   font-lock-variable-name-face)
   (cons p4_14-primitives-regexp  font-lock-function-name-face)
   (cons p4_14-operations-regexp  font-lock-builtin-face)
   (cons p4_14-keywords-regexp    font-lock-keyword-face)
   (cons p4_14-pragma-regexp      font-lock-keyword-face)
   (cons "\\(\\w*_t +\\)"         font-lock-type-face)
   (cons "\\(<[^>]+>\\)"          font-lock-string-face)
   (cons "[^_A-Za-z0-9]\\(0x[0-9A-Fa-f]+\\)" '(1  font-lock-constant-face))
   (cons "[^_A-Za-z0-9]\\(0b[01]+\\)"        '(1  font-lock-constant-face))
   (cons "[^_A-Za-z0-9]\\([+-]?[0-9]+\\)"    '(1  font-lock-constant-face))
;;   (cons "\\(\\w*\\)"        font-lock-variable-name-face)
   )
  "Default Highlighting Expressions for P4_14")

(defconst p4_14-font-lock-syntax-alist
  (list
   )
  )

(defvar p4_14-mode-syntax-table
  (let ((st (make-syntax-table)))
    (modify-syntax-entry  ?_  "w"      st)
    (modify-syntax-entry  ?/  ". 124b" st)
    (modify-syntax-entry  ?*  ". 23"   st)
    (modify-syntax-entry  ?\n  "> b"   st)
    st)
  "Syntax table for p4_14-mode")

;;; Indentation
(defvar p4_14-indent-offset 4
  "Indentation offset for `p4_14-mode'.")

(defun p4_14-indent-line ()
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
              (setq indent-col (+ indent-col p4_14-indent-offset))))
        (error nil)))
    (save-excursion
      (back-to-indentation)
      (when (and (looking-at indentation-decreasers)
                 (>= indent-col p4_14-indent-offset))
        (setq indent-col (- indent-col p4_14-indent-offset))))
    (indent-line-to indent-col)))

;;; Imenu support
(require 'imenu)
(setq p4_14-imenu-generic-expression
      '(
        ("Controls"      "^control *\\([A-Za-z0-9_]*\\)"      1)
        ("Externs"       "^blackbox *\\([A-Za-z0-9_]*\\) *\\([A-Za-z0-9_]*\\)" 2)
        ("Registers"     "^register *\\([A-Za-z0-9_]*\\)"     1)
        ("Meters"        "^meter *\\([A-Za-z0-9_]*\\)"        1)
        ("Counters"      "^counter *\\([A-Za-z0-9_]*\\)"      1)
        ("Calculations"  "^field_list_calculation *\\([A-Za-z0-9_]*\\)"   1)
        ("Field Lists"   "^field_list  *\\([A-Za-z0-9_]*\\)"  1)
        ("Tables"        "^table *\\([A-Za-z0-9_]*\\)"        1)
        ("Actions"       "^action *\\([A-Za-z0-9_]*\\)"       1)
        ("Parser States" "^parser *\\([A-Za-z0-9_]*\\)"       1)
        ("Header Types"  "^header_type *\\([A-Za-z0-9_]*\\)"  1)
        ))

;;; Cscope Support
(require 'xcscope)

;; Put everything together
(defun p4_14-mode ()
  "Major mode for editing P4_14 programs"
  (interactive)
  (kill-all-local-variables)
  (set-syntax-table p4_14-mode-syntax-table)
  (use-local-map p4_14-mode-map)
  (set (make-local-variable 'font-lock-defaults) '(p4_14-font-lock-keywords))
  (set (make-local-variable 'indent-line-function) 'p4_14-indent-line)  
  (setq major-mode 'p4_14-mode)
  (setq mode-name "P4_14")
  (setq imenu-generic-expression p4_14-imenu-generic-expression)
  (imenu-add-to-menubar "P4_14")
  (cscope-minor-mode)
  (run-hooks 'p4_14-mode-hook)
)

;; The most important line
(provide 'p4_14-mode)
