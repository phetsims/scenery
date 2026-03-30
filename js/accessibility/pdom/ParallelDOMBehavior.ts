// Copyright 2026, University of Colorado Boulder

/**
 * Helper utilities for evaluating accessibility behavior functions and applying fast-path peer updates.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import type Node from '../../nodes/Node.js';
import type { ParallelDOMOptions, PDOMValueType } from './ParallelDOM.js';
import type PDOMInstance from './PDOMInstance.js';

export type PDOMBehaviorFunction<AllowedKeys extends keyof ParallelDOMOptions> = ( node: Node, options: Pick<ParallelDOMOptions, AllowedKeys>, value: PDOMValueType, callbacksForOtherNodes: ( () => void )[] ) => ParallelDOMOptions;

// Behavior option keys that can be applied by patching existing peer content/attributes without rebuilding peers.
// Any additional defined key from accessibleNameBehavior requires a full PDOM render.
// WARNING: If you add keys here, you must also update applyAccessibleNameFastPath() (and related tests) so every
// declared fast-path key has concrete peer patch logic.
export const ACCESSIBLE_NAME_FAST_PATH_KEYS: readonly ( keyof ParallelDOMOptions )[] = [ 'innerContent', 'ariaLabel', 'labelContent' ];

// Behavior option keys that can be applied directly by updating paragraph sibling content.
// Any additional defined key from accessibleParagraphBehavior requires a full PDOM render.
// WARNING: If you add keys here, update applyAccessibleParagraphFastPath() so every declared key is handled.
export const ACCESSIBLE_PARAGRAPH_FAST_PATH_KEYS: readonly ( keyof ParallelDOMOptions )[] = [ 'accessibleParagraphContent' ];

// Behavior option keys that can be applied directly by updating description sibling content.
// Any additional defined key from accessibleHelpTextBehavior requires a full PDOM render.
// WARNING: If you add keys here, update applyAccessibleHelpTextFastPath() so every declared key is handled.
export const ACCESSIBLE_HELP_TEXT_FAST_PATH_KEYS: readonly ( keyof ParallelDOMOptions )[] = [ 'descriptionContent' ];

/**
 * Evaluates a behavior function with per-call scratch objects, then runs callback logic with resulting options.
 */
export const evaluateBehaviorOptions = <AllowedKeys extends keyof ParallelDOMOptions>(
  node: Node,
  behaviorFunction: PDOMBehaviorFunction<AllowedKeys>,
  value: PDOMValueType,
  callback: ( behaviorOptions: ParallelDOMOptions, hasOtherNodeCallbacks: boolean ) => void
): void => {
  const behaviorOptionsScratch = {} as Pick<ParallelDOMOptions, AllowedKeys>;
  const callbacksScratch: ( () => void )[] = [];
  const behaviorOptions = behaviorFunction(
    node,
    behaviorOptionsScratch,
    value,
    callbacksScratch
  );
  callback( behaviorOptions, callbacksScratch.length > 0 );
};

/**
 * Returns whether a behavior result requires a full PDOM re-render instead of a fast peer-level update.
 * Re-render is required if:
 * - forceRender is true,
 * - callbacks for other Nodes are present, or
 * - any defined option is outside the allowed fast-path keys.
 * Optionally, defined keys in ignoredDefinedKeys are skipped for the render check.
 * Optionally requires that at least one allowed key is defined.
 */
export const behaviorOptionsRequireRender = (
  behaviorOptions: ParallelDOMOptions,
  allowedFastPathKeys: readonly ( keyof ParallelDOMOptions )[],
  hasOtherNodeCallbacks: boolean,
  forceRender: boolean,
  requireAtLeastOneAllowedKey: boolean,
  ignoredDefinedKeys?: ReadonlySet<keyof ParallelDOMOptions>
): boolean => {

  if ( forceRender || hasOtherNodeCallbacks ) {
    return true;
  }

  let hasAllowedFastPathOption = false;
  for ( const key in behaviorOptions ) {
    const optionKey = key as keyof ParallelDOMOptions;
    const optionValue = behaviorOptions[ optionKey ];

    if ( optionValue === undefined ) {
      continue;
    }
    if ( ignoredDefinedKeys?.has( optionKey ) ) {
      continue;
    }
    if ( allowedFastPathKeys.includes( optionKey ) ) {
      hasAllowedFastPathOption = true;
    }
    else {
      return true;
    }
  }

  return requireAtLeastOneAllowedKey && !hasAllowedFastPathOption;
};

/**
 * Returns whether a behavior-provided labelTagName can be treated as non-structural for this update.
 *
 * This is true only when all existing peers already have a label sibling with the same tag name.
 * If any peer is missing a label sibling, or the tag differs, a full render is required to rebuild structure.
 */
export const isFastPathSafeLabelTagName = ( labelTagName: string | null, pdomInstances: PDOMInstance[] ): boolean => {
  if ( typeof labelTagName !== 'string' ) {
    return false;
  }

  for ( let i = 0; i < pdomInstances.length; i++ ) {
    const labelSibling = pdomInstances[ i ].peer!.getLabelSibling();
    if ( !labelSibling || labelSibling.tagName.toLowerCase() !== labelTagName.toLowerCase() ) {
      return false;
    }
  }

  return true;
};

/**
 * Returns whether a behavior-provided descriptionTagName can be treated as non-structural for this update.
 *
 * This is true only when all existing peers already have a description sibling with the same tag name.
 * If any peer is missing a description sibling, or the tag differs, a full render is required to rebuild structure.
 */
export const isFastPathSafeDescriptionTagName = ( descriptionTagName: string | null, pdomInstances: PDOMInstance[] ): boolean => {
  if ( typeof descriptionTagName !== 'string' ) {
    return false;
  }

  for ( let i = 0; i < pdomInstances.length; i++ ) {
    const descriptionSibling = pdomInstances[ i ].peer!.getDescriptionSibling();
    if ( !descriptionSibling || descriptionSibling.tagName.toLowerCase() !== descriptionTagName.toLowerCase() ) {
      return false;
    }
  }

  return true;
};

/**
 * Returns whether a behavior-provided appendLabel can be treated as non-structural for this update.
 *
 * This is true only when all existing peers already have label siblings ordered relative to the primary
 * sibling in the same way that appendLabel would require.
 */
export const isFastPathSafeAppendLabel = ( appendLabel: boolean, pdomInstances: PDOMInstance[] ): boolean => {
  if ( typeof appendLabel !== 'boolean' ) {
    return false;
  }

  for ( let i = 0; i < pdomInstances.length; i++ ) {
    const peer = pdomInstances[ i ].peer!;
    const labelSibling = peer.getLabelSibling();
    const primarySibling = peer.getPrimarySibling();

    if ( !labelSibling || !primarySibling ) {
      return false;
    }

    const parentElement = labelSibling.parentElement;
    if ( !parentElement || primarySibling.parentElement !== parentElement ) {
      return false;
    }

    const labelIndex = Array.prototype.indexOf.call( parentElement.children, labelSibling );
    const primaryIndex = Array.prototype.indexOf.call( parentElement.children, primarySibling );
    if ( labelIndex === -1 || primaryIndex === -1 ) {
      return false;
    }

    if ( appendLabel && labelIndex < primaryIndex ) {
      return false;
    }
    if ( !appendLabel && labelIndex > primaryIndex ) {
      return false;
    }
  }

  return true;
};

/**
 * Returns whether a behavior-provided appendDescription can be treated as non-structural for this update.
 *
 * This is true only when all existing peers already have description siblings ordered relative to the primary
 * sibling in the same way that appendDescription would require.
 */
export const isFastPathSafeAppendDescription = ( appendDescription: boolean, pdomInstances: PDOMInstance[] ): boolean => {
  if ( typeof appendDescription !== 'boolean' ) {
    return false;
  }

  for ( let i = 0; i < pdomInstances.length; i++ ) {
    const peer = pdomInstances[ i ].peer!;
    const descriptionSibling = peer.getDescriptionSibling();
    const primarySibling = peer.getPrimarySibling();

    if ( !descriptionSibling || !primarySibling ) {
      return false;
    }

    const parentElement = descriptionSibling.parentElement;
    if ( !parentElement || primarySibling.parentElement !== parentElement ) {
      return false;
    }

    const descriptionIndex = Array.prototype.indexOf.call( parentElement.children, descriptionSibling );
    const primaryIndex = Array.prototype.indexOf.call( parentElement.children, primarySibling );
    if ( descriptionIndex === -1 || primaryIndex === -1 ) {
      return false;
    }

    if ( appendDescription && descriptionIndex < primaryIndex ) {
      return false;
    }
    if ( !appendDescription && descriptionIndex > primaryIndex ) {
      return false;
    }
  }

  return true;
};

/**
 * Applies behavior-produced accessibleName options directly to existing peers without rebuilding sibling structure.
 */
export const applyAccessibleNameFastPath = (
  pdomInstances: PDOMInstance[],
  behaviorOptions: ParallelDOMOptions,
  unwrapValue: ( valueOrProperty: PDOMValueType ) => string | null
): void => {
  const innerContent = behaviorOptions.innerContent === undefined ? undefined : unwrapValue( behaviorOptions.innerContent );
  const labelContent = behaviorOptions.labelContent === undefined ? undefined : unwrapValue( behaviorOptions.labelContent );
  const ariaLabel = behaviorOptions.ariaLabel === undefined ? undefined : unwrapValue( behaviorOptions.ariaLabel );

  for ( let i = 0; i < pdomInstances.length; i++ ) {
    const peer = pdomInstances[ i ].peer!;

    if ( innerContent !== undefined ) {
      peer.setPrimarySiblingContent( innerContent );
    }
    if ( labelContent !== undefined ) {
      peer.setLabelSiblingContent( labelContent );
    }
    if ( ariaLabel !== undefined ) {
      const primarySibling = peer.getPrimarySibling();
      if ( primarySibling ) {
        if ( ariaLabel === null ) {
          primarySibling.removeAttribute( 'aria-label' );
        }
        else {

          // Apply aria-label directly to the peer for fast-path updates. We intentionally avoid mutating
          // model-level attribute state (like setPDOMAttribute/setAriaLabel) so behavior output stays
          // render-time only and base options are restored correctly when accessibleName is cleared.
          peer.setAttributeToElement( 'aria-label', ariaLabel );
        }
      }
    }
  }
};

/**
 * Applies accessible paragraph content directly to existing peers without rebuilding sibling structure.
 */
export const applyAccessibleParagraphFastPath = ( pdomInstances: PDOMInstance[], content: string | null ): void => {
  for ( let i = 0; i < pdomInstances.length; i++ ) {
    const peer = pdomInstances[ i ].peer!;
    peer.setAccessibleParagraphContent( content );
  }
};

/**
 * Applies behavior-produced accessibleHelpText options directly to existing peers without rebuilding sibling structure.
 */
export const applyAccessibleHelpTextFastPath = (
  pdomInstances: PDOMInstance[],
  behaviorOptions: ParallelDOMOptions,
  unwrapValue: ( valueOrProperty: PDOMValueType ) => string | null
): void => {
  const descriptionContent = behaviorOptions.descriptionContent === undefined ? undefined : unwrapValue( behaviorOptions.descriptionContent );

  if ( descriptionContent === undefined ) {
    return;
  }

  for ( let i = 0; i < pdomInstances.length; i++ ) {
    const peer = pdomInstances[ i ].peer!;
    peer.setDescriptionSiblingContent( descriptionContent );
  }
};