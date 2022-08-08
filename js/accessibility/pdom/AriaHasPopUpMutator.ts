// Copyright 2021-2022, University of Colorado Boulder

/**
 * Sets ParallelDOM.js fields for an Node that uses aria-haspopup. It was discovered that
 * this attribute changes how iOS 14 VoiceOver interacts with elements - elements with
 * aria-haspopup must be positioned in the viewport to receive pointer and click
 * events.
 *
 * See https://github.com/phetsims/scenery/issues/1094 for more information.
 *
 * @author Jesse Greenberg
 */
import { Node, scenery } from '../../imports.js';

const AriaHasPopUpMutator = {

  /**
   * @param node - Node whose ParallelDOM.js fields will change
   * @param value - Valid value for aria-haspopup attribute, or false to remove the attribute
   */
  mutateNode( node: Node, value: boolean | string ): void {
    if ( value ) {
      node.setPDOMAttribute( 'aria-haspopup', value );
    }
    else {
      assert && assert( node.hasPDOMAttribute( 'aria-haspopup' ), 'Set aria-haspopup once before removing it.' );
      node.removePDOMAttribute( 'aria-haspopup' );
    }

    node.positionInPDOM = !!value;
  }
};

scenery.register( 'AriaHasPopUpMutator', AriaHasPopUpMutator );
export default AriaHasPopUpMutator;