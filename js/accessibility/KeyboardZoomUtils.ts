// Copyright 2019-2025, University of Colorado Boulder

/**
 * Utilities specific to the keyboard for handling zoom/pan control.
 *
 * @author Jesse Greenberg
 */

import Property from '../../../axon/js/Property.js';
import HotkeyData from '../input/HotkeyData.js';
import KeyboardUtils from '../accessibility/KeyboardUtils.js';
import scenery from '../scenery.js';

const KeyboardZoomUtils = {

  /**
   * Returns true if the platform is most likely a Mac device. Pan/Zoom will use different modifier keys in this case.
   *
   * TODO: Move to platform if generally useful? https://github.com/phetsims/scenery/issues/1581
   */
  isPlatformMac: (): boolean => {
    return _.includes( window.navigator.platform, 'Mac' );
  },

  /**
   * Get the 'meta' key for the platform that would indicate user wants to zoom. This is 'metaKey' on Mac and 'ctrl'
   * on Windows.
   *
   */
  getPlatformZoomMetaKey: (): string => {
    return KeyboardZoomUtils.isPlatformMac() ? 'metaKey' : 'ctrlKey';
  },

  /**
   * Returns true of the keyboard input indicates that a zoom command was initiated. Different keys are checked
   * on mac devices (which go through the Cmd key) and windows devices (which use the ctrl modifier).
   *
   * @param event
   * @param zoomIn - do you want to check for zoom in or zoom out?
   */
  isZoomCommand: ( event: Event, zoomIn: boolean ): boolean => {

    // This function checks the meta key on the event, so it cannot use HotkeyData.
    const zoomKey = zoomIn ? KeyboardUtils.KEY_EQUALS : KeyboardUtils.KEY_MINUS;
    const metaKey = KeyboardZoomUtils.getPlatformZoomMetaKey();

    // @ts-expect-error
    return event[ metaKey ] && KeyboardUtils.isKeyEvent( event, zoomKey );
  },

  /**
   * Returns true if the keyboard command indicates a "zoom reset". This is ctrl + 0 on Win and cmd + 0 on mac.
   */
  isZoomResetCommand: ( event: Event ): boolean => {
    const metaKey = KeyboardZoomUtils.getPlatformZoomMetaKey();

    // This function uses the meta key on the event, so it cannot use HotkeyData.
    // @ts-expect-error
    return event[ metaKey ] && KeyboardUtils.isKeyEvent( event, KeyboardUtils.KEY_0 );
  },

  // Hotkey data is not used in the implementation but is provided for documentation purposes.
  // Beware if you change keys in these, you will need to change other methods in this utils file.
  ZOOM_IN_HOTKEY_DATA: new HotkeyData( {
    keyStringProperties: [ new Property( 'ctrl+equals' ), new Property( 'meta+equals' ) ],
    binderName: 'Zoom in',
    repoName: 'scenery',
    global: true
  } ),

  ZOOM_OUT_HOTKEY_DATA: new HotkeyData( {
    keyStringProperties: [ new Property( 'ctrl+minus' ), new Property( 'meta+minus' ) ],
    binderName: 'Zoom in',
    repoName: 'scenery',
    global: true
  } ),

  RESET_ZOOM_HOTKEY_DATA: new HotkeyData( {
    keyStringProperties: [ new Property( 'ctrl+0' ), new Property( 'meta+0' ) ],
    binderName: 'Reset zoom',
    repoName: 'scenery',
    global: true
  } )
};

scenery.register( 'KeyboardZoomUtils', KeyboardZoomUtils );
export default KeyboardZoomUtils;