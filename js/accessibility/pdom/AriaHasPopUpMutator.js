// Copyright 2020, University of Colorado Boulder

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
import scenery from '../../scenery.js';

const AriaHasPopUpMutator = {

  /**
   * @public
   * @param {Node} node - Node whose ParallelDOM.js fields will change
   * @param {boolean} hasPopUp
   */
  mutateNode( node, hasPopUp ) {
    if ( hasPopUp ) {
      node.setAccessibleAttribute( 'aria-haspopup', true );
    }
    else {
      assert && assert( node.hasAccessibleAttribute( 'aria-haspopup' ), 'Set aria-haspopup once before removing it.' );
      node.removeAccessibleAttribute( 'aria-haspopup' );
    }

    node.positionSiblings = hasPopUp;
  }
};

scenery.register( 'AriaHasPopUpMutator', AriaHasPopUpMutator );
export default AriaHasPopUpMutator;